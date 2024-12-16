import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # 导入tqdm
import numpy as np
from datetime import datetime

# 注册tqdm与pandas的progress_apply接口
tqdm.pandas()

dataFile = 'data.txt'

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clustering_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

try:
    logger.info("开始执行聚类分析脚本。")

    # 指定临时目录到 D 盘，避免C盘空间不足
    os.environ['TMP'] = 'D:\\temp'
    os.environ['TEMP'] = 'D:\\temp'
    logger.info("已设置临时目录到 D 盘。")

    # 读取数据
    column_names = [
        'day_id', 'calling_nbr', 'called_nbr', 'calling_optr', 'called_optr',
        'calling_city', 'called_city', 'calling_roam_city', 'called_roam_city',
        'start_time', 'end_time', 'raw_dur', 'call_type', 'calling_cell'
    ]

    logger.info("开始读取数据文件。")
    df = pd.read_csv(dataFile, sep='\t', header=None, names=column_names)
    logger.info(f"数据读取完成，共有 {df.shape[0]} 行，{df.shape[1]} 列。")

    # 数据预处理
    logger.info("开始数据预处理。")
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time
    df['raw_dur'] = df['raw_dur'].astype(int)
    logger.info("数据预处理完成。")

    # 检查并处理缺失值
    if df.isnull().values.any():
        logger.warning("数据中存在缺失值，正在填补。")
        df = df.fillna(0)

    # 检查异常值（极端高的通话时长）
    logger.info("检查数据中的异常值。")
    dur_threshold = df['raw_dur'].quantile(0.99)
    initial_shape = df.shape
    df = df[df['raw_dur'] <= dur_threshold]
    logger.info(f"删除了 raw_dur > {dur_threshold} 的异常值，原始形状 {initial_shape}，处理后 {df.shape}。")

    # 构建特征1：通话呼叫行为
    logger.info("开始构建特征1：通话呼叫行为。")
    caller_counts = df.groupby('calling_nbr').size().rename('caller_count')
    callee_counts = df.groupby('called_nbr').size().rename('callee_count')
    features = pd.concat([caller_counts, callee_counts], axis=1).fillna(0)
    logger.info("特征1构建完成。")

    # 特征2：通话时刻
    logger.info("开始构建特征2：通话时刻。")
    df['start_time_seconds'] = df['start_time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    bins = [0, 6*3600, 12*3600, 18*3600, 24*3600]
    labels = ['Early Morning', 'Morning', 'Afternoon', 'Evening']
    df['time_period'] = pd.cut(df['start_time_seconds'], bins=bins, labels=labels, right=False)
    time_period_counts = pd.get_dummies(df['time_period']).groupby(df['calling_nbr']).sum()
    features = features.join(time_period_counts, how='left').fillna(0)
    logger.info("特征2构建完成。")

    # 特征3：通话时长
    logger.info("开始构建特征3：通话时长。")
    duration_stats = df.groupby('calling_nbr')['raw_dur'].agg(['sum','mean','max','min','count']).rename(
        columns={'sum':'total_dur','mean':'avg_dur','max':'max_dur','min':'min_dur','count':'call_count'}
    )
    features = features.join(duration_stats, how='left').fillna(0)
    logger.info("特征3构建完成。")

    # 特征4：通话类型分布和运营商比例
    logger.info("开始构建特征4：通话类型分布和运营商比例。")
    call_type_dummies = pd.get_dummies(df['call_type'], prefix='call_type')
    call_type_counts = call_type_dummies.groupby(df['calling_nbr']).sum()
    features = features.join(call_type_counts, how='left').fillna(0)
    logger.info("通话类型分布特征构建完成。")

    called_optr_dummies = pd.get_dummies(df['called_optr'], prefix='optr')
    called_optr_counts = called_optr_dummies.groupby(df['calling_nbr']).sum()
    features = features.join(called_optr_counts, how='left').fillna(0)
    logger.info("不同运营商通话次数特征构建完成。")

    # 运营商比例
    features['optr_1_ratio'] = features['optr_1'] / features['caller_count']
    features['optr_2_ratio'] = features['optr_2'] / features['caller_count']
    features['optr_3_ratio'] = features['optr_3'] / features['caller_count']

    features = features.fillna(0)
    logger.info("所有特征构建完成。")
    print(features.head())

    # 选择数值型特征
    logger.info("选择数值型特征进行聚类。")
    selected_features = [
        'caller_count', 'callee_count', 'Early Morning', 'Morning',
        'Afternoon', 'Evening', 'total_dur', 'avg_dur', 'max_dur',
        'min_dur', 'call_count', 'call_type_1', 'call_type_2',
        'call_type_3', 'optr_1_ratio', 'optr_2_ratio', 'optr_3_ratio'
    ]
    X = features[selected_features]
    logger.info(f"特征数据形状: {X.shape}")

    features = features.reset_index(drop=True)
    logger.info("重置特征数据框的索引。")

    # 标准化
    logger.info("开始进行特征归一化。")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    logger.info("特征归一化完成。")

    # PCA降维，减少计算量（例如降到min(X.shape[1], 50)维）
    logger.info("PCA降维")
    n_components = min(X_scaled.shape[1], 50)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA降维到 {n_components} 维完成。")

    # 使用MiniBatchKMeans替代KMeans，加快大数据聚类速度
    from sklearn.cluster import MiniBatchKMeans

    # 使用肘部法则确定k值（仍在全量数据上计算SSE）
    logger.info("开始使用肘部法则确定最佳聚类数。")
    K = range(1, 11)
    sse = []
    for k in tqdm(K, desc="Elbow Method Progress"):
        tqdm.write(f"正在计算 k={k} 的MiniBatchKMeans聚类...")
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=1000,  # 可根据数据量调整
            n_init=5,
            max_iter=100
        )
        mbk.fit(X_pca)
        sse.append(mbk.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, sse, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('elbow_method.png')
    logger.info("肘部法则图已保存为 elbow_method.png。")
    plt.show()

    # 为计算轮廓系数进行抽样
    logger.info("为计算轮廓系数进行抽样。")
    sample_fraction = 0.05  # 根据数据大小调整
    sample_size = int(X_pca.shape[0] * sample_fraction)
    sample_size = min(sample_size, 10000)  # 限制最大抽样数
    np.random.seed(42)
    indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
    X_sample = X_pca[indices]

    # 使用轮廓系数确定最佳k值（在抽样数据上计算）
    logger.info("开始使用轮廓系数确定最佳聚类数（基于抽样数据）。")
    silhouette_scores = []
    for k in tqdm(range(2, 11), desc="Silhouette Score Progress"):
        tqdm.write(f"正在计算 k={k} 的轮廓系数（基于抽样数据）...")
        try:
            mbk = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                batch_size=1000,
                n_init=5,
                max_iter=100
            )
            labels_sample = mbk.fit_predict(X_sample)
            score = silhouette_score(X_sample, labels_sample)
            silhouette_scores.append(score)
            tqdm.write(f"k={k}, 轮廓系数={score:.4f}")
        except Exception as e:
            tqdm.write(f"k={k}, 计算失败: {e}")
            silhouette_scores.append(None)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k (Sampled Data)')
    plt.savefig('silhouette_analysis.png')
    logger.info("轮廓系数图已保存为 silhouette_analysis.png。")
    plt.show()

    valid_silhouette_scores = [s for s in silhouette_scores if s is not None]
    valid_k = list(range(2, 11))
    if len(valid_silhouette_scores) > 0:
        best_k_silhouette = valid_k[np.argmax(valid_silhouette_scores)]
        best_score = max(valid_silhouette_scores)
        logger.info(f"基于抽样数据的轮廓系数选择 k={best_k_silhouette}, 得分={best_score:.4f}")
    else:
        best_k_silhouette = None
        logger.warning("没有有效轮廓系数，使用默认k=4。")

    optimal_k = best_k_silhouette if best_k_silhouette is not None else 4
    logger.info(f"最终选择的最佳k={optimal_k}。")

    # 使用最终k对全量数据聚类
    logger.info(f"使用 k={optimal_k} 对全量数据进行MiniBatchKMeans聚类。")
    mbk_final = MiniBatchKMeans(
        n_clusters=optimal_k,
        random_state=42,
        batch_size=1000,
        n_init=5,
        max_iter=100
    )
    mbk_final.fit(X_pca)
    labels = mbk_final.labels_
    logger.info("全量数据聚类完成。")

    # 检查标签长度
    logger.info(f"K-Means标签数量: {len(labels)}, 特征数据框行数: {features.shape[0]}")
    if len(labels) == features.shape[0]:
        features['cluster'] = labels
        logger.info("聚类标签已成功添加到特征数据框中。")
    else:
        logger.error("聚类标签数量与特征数据框行数不一致，无法赋值。")
        features['cluster'] = None

    print(features['cluster'].head())
    logger.info("显示部分聚类标签数据。")

    # 聚类中心（逆标准化）
    logger.info("恢复聚类中心(逆PCA+逆标准化)。")
    cluster_centers_pca = mbk_final.cluster_centers_
    cluster_centers_scaled = pca.inverse_transform(cluster_centers_pca)
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=selected_features)
    cluster_centers_df.index = [f'Cluster {i}' for i in range(optimal_k)]
    print(cluster_centers_df)
    logger.info("聚类中心（逆标准化）如下：")
    logger.info(f"\n{cluster_centers_df}")

    # 可视化前两个主成分
    logger.info("PCA降维后可视化聚类结果（前两维）。")
    pca_vis = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_vis = pca_vis.iloc[:, :2]
    pca_vis['cluster'] = labels.astype(str)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_vis, x='PC1', y='PC2', hue='cluster', palette='Set1', alpha=0.6)
    plt.title(f'MiniBatchKMeans Clustering (k={optimal_k})')
    plt.savefig('kmeans_pca_scatter.png')
    logger.info("聚类结果散点图已保存为 kmeans_pca_scatter.png。")
    plt.show()

    logger.info("聚类分析完成。")

except Exception as e:
    logger.exception("在执行聚类分析脚本时发生错误。")