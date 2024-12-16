import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 注册tqdm与pandas的progress_apply接口
tqdm.pandas()

dataFile = 'data.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clustering_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

try:
    logger.info("开始执行聚类分析脚本。")

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
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time
    df['raw_dur'] = df['raw_dur'].astype(int)

    if df.isnull().values.any():
        logger.warning("数据中存在缺失值，使用0填补。")
        df = df.fillna(0)

    # 删除极端值
    dur_threshold = df['raw_dur'].quantile(0.99)
    df = df[df['raw_dur'] <= dur_threshold]

    # 构建特征（与之前相同的特征构建步骤）
    caller_counts = df.groupby('calling_nbr').size().rename('caller_count')
    callee_counts = df.groupby('called_nbr').size().rename('callee_count')
    features = pd.concat([caller_counts, callee_counts], axis=1).fillna(0)

    df['start_time_seconds'] = df['start_time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    bins = [0, 6*3600, 12*3600, 18*3600, 24*3600]
    labels = ['Early Morning', 'Morning', 'Afternoon', 'Evening']
    df['time_period'] = pd.cut(df['start_time_seconds'], bins=bins, labels=labels, right=False)
    time_period_counts = pd.get_dummies(df['time_period']).groupby(df['calling_nbr']).sum()
    features = features.join(time_period_counts, how='left').fillna(0)

    duration_stats = df.groupby('calling_nbr')['raw_dur'].agg(['sum','mean','max','min','count']).rename(
        columns={'sum':'total_dur','mean':'avg_dur','max':'max_dur','min':'min_dur','count':'call_count'}
    )
    features = features.join(duration_stats, how='left').fillna(0)

    call_type_dummies = pd.get_dummies(df['call_type'], prefix='call_type')
    call_type_counts = call_type_dummies.groupby(df['calling_nbr']).sum()
    features = features.join(call_type_counts, how='left').fillna(0)

    called_optr_dummies = pd.get_dummies(df['called_optr'], prefix='optr')
    called_optr_counts = called_optr_dummies.groupby(df['calling_nbr']).sum()
    features = features.join(called_optr_counts, how='left').fillna(0)

    features['optr_1_ratio'] = features['optr_1'] / features['caller_count']
    features['optr_2_ratio'] = features['optr_2'] / features['caller_count']
    features['optr_3_ratio'] = features['optr_3'] / features['caller_count']

    features = features.fillna(0)
    logger.info("特征构建完成。")

    # 选择特征列
    selected_features = [
        'caller_count', 'callee_count', 'Early Morning', 'Morning',
        'Afternoon', 'Evening', 'total_dur', 'avg_dur', 'max_dur',
        'min_dur', 'call_count', 'call_type_1', 'call_type_2',
        'call_type_3', 'optr_1_ratio', 'optr_2_ratio', 'optr_3_ratio'
    ]
    X = features[selected_features]
    features = features.reset_index(drop=True)

    # 使用 MinMaxScaler 而非 StandardScaler
    logger.info("使用MinMaxScaler归一化。")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维
    logger.info("PCA降维")
    n_components = min(X.shape[1], 20)  # 降到20维试试，比50更小，进一步加快
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    logger.info("尝试使用高斯混合模型(GaussianMixture)聚类。")

    # 使用轮廓系数选择n_components(类似于选择k)
    # 尝试不同的n_components（类似k值）
    n_range = range(2, 11)
    silhouette_scores = []

    # 抽样
    logger.info("抽样数据用于计算轮廓系数。")
    sample_fraction = 0.05
    sample_size = int(X_pca.shape[0] * sample_fraction)
    sample_size = min(sample_size, 5000) # 更小的抽样上限
    np.random.seed(42)
    indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
    X_sample = X_pca[indices]

    from sklearn.mixture import GaussianMixture
    logger.info("开始使用GaussianMixture进行聚类，并在抽样数据上评估轮廓系数。")

    for n_comp in tqdm(n_range, desc="GMM Components Progress"):
        tqdm.write(f"正在计算 n_components={n_comp} 的轮廓系数(抽样数据)...")
        try:
            gmm = GaussianMixture(n_components=n_comp, random_state=42)
            gmm.fit(X_sample)
            labels_sample = gmm.predict(X_sample)
            score = silhouette_score(X_sample, labels_sample)
            silhouette_scores.append(score)
            tqdm.write(f"n_components={n_comp}, 轮廓系数={score:.4f}")
        except Exception as e:
            tqdm.write(f"n_components={n_comp}, 计算失败: {e}")
            silhouette_scores.append(None)

    valid_scores = [(nc, sc) for nc, sc in zip(n_range, silhouette_scores) if sc is not None]
    if valid_scores:
        best_nc, best_score = max(valid_scores, key=lambda x: x[1])
        logger.info(f"基于抽样数据的轮廓系数选择的最佳 n_components={best_nc}, 得分={best_score:.4f}")
    else:
        best_nc = 4
        logger.warning("没有有效的轮廓系数，使用默认n_components=4")

    logger.info(f"最终选择 n_components={best_nc} 进行全量聚类。")

    # 使用最佳n_components对全量数据进行聚类
    gmm_final = GaussianMixture(n_components=best_nc, random_state=42)
    gmm_final.fit(X_pca)
    labels = gmm_final.predict(X_pca)

    if len(labels) == features.shape[0]:
        features['cluster'] = labels
        logger.info("聚类标签已加入特征数据框。")
    else:
        logger.error("标签数量与特征数据框行数不一致，无法赋值。")
        features['cluster'] = None

    print(features['cluster'].head())
    logger.info("显示部分聚类标签数据。")

    # 恢复聚类中心对于GMM不一样，GMM中没有像KMeans那样的簇中心，
    # 而是有高斯分布的均值。可以提取gmm_final.means_作为“中心”
    # 逆PCA+逆缩放恢复原特征空间下的均值
    cluster_centers_pca = gmm_final.means_
    cluster_centers_scaled = pca.inverse_transform(cluster_centers_pca)
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=selected_features)
    cluster_centers_df.index = [f'Cluster {i}' for i in range(best_nc)]
    print(cluster_centers_df)
    logger.info("GMM均值（逆标准化后）：")
    logger.info(f"\n{cluster_centers_df}")

    # 可视化前两个主成分
    logger.info("可视化GMM聚类结果（前两维）。")
    pca_vis = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_vis = pca_vis.iloc[:, :2]
    pca_vis['cluster'] = labels.astype(str)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_vis, x='PC1', y='PC2', hue='cluster', palette='Set1', alpha=0.6)
    plt.title(f'GMM Clustering (n_components={best_nc})')
    plt.savefig('gmm_pca_scatter.png')
    logger.info("聚类结果散点图已保存为 gmm_pca_scatter.png。")
    plt.show()

    logger.info("聚类分析完成。")

except Exception as e:
    logger.exception("在执行聚类分析脚本时发生错误。")
