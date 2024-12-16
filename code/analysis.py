import pandas as pd
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
import logging
import os
import gc

# 指定临时目录到 D 盘，避免C盘空间不足
os.environ['TMP'] = 'D:\\temp'
os.environ['TEMP'] = 'D:\\temp'

# 设置日志以实时监控代码执行
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 1. 读取数据
data = pd.read_csv('data.txt', sep='\t', header=None,
                   dtype={'calling_nbr': str,
                          'called_nbr': str,
                          'calling_city': str,
                          'called_city': str,
                          'calling_roam_city': str,
                          'called_roam_city': str,
                          'calling_cell': str})

# 2. 数据预处理与特征工程

logger.info("开始数据预处理与特征工程...")

# 2.1 将通话开始时间转换为小时
logger.info("转换通话开始时间为小时...")
data['start_hour'] = pd.to_datetime(data['start_time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour

# 检查是否有无法转换的时间
if data['start_hour'].isnull().any():
    logger.warning("存在无法转换的 start_time，相关行将被移除。")
    data = data.dropna(subset=['start_hour'])

# 2.2 将24小时分为8个时间段，每3小时一段 (0-3,3-6,...,21-24)
logger.info("划分时间段...")
data['time_slot'] = data['start_hour'] // 3 + 1

# 2.3 通话时长转换为数值类型
logger.info("提取并转换通话时长为数值类型...")
data['call_duration'] = pd.to_numeric(data['raw_dur'], errors='coerce')

# 检查并处理无法转换的通话时长
if data['call_duration'].isnull().any():
    logger.warning("存在无法转换的 call_duration，相关行将被移除。")
    data = data.dropna(subset=['call_duration'])

# 2.4 用户级别聚合特征（平均通话时长和总通话次数）
logger.info("聚合用户级别特征（平均通话时长和总通话次数）...")
user_features = data.groupby('calling_nbr').agg(
    avg_call_duration=('call_duration', 'mean'),
    total_calls=('call_duration', 'count')
).reset_index()

# 2.5 计算各时间段通话比例
logger.info("计算各时间段通话比例...")
time_slot_counts = data.groupby(['calling_nbr', 'time_slot']).size().unstack(fill_value=0)

# 重命名时间段列名为字符串，避免混合类型
logger.info("重命名时间段列名为字符串...")
time_slot_counts.columns = [f"time_slot_{int(col)}" for col in time_slot_counts.columns]

# 计算时间段比例
logger.info("计算时间段通话比例...")
time_slot_props = time_slot_counts.div(time_slot_counts.sum(axis=1), axis=0).reset_index()

# 2.6 合并用户级别特征（平均通话时长、总通话数、时间段分布）
logger.info("合并用户级别特征...")
user_data = pd.merge(user_features, time_slot_props, on='calling_nbr', how='left')

# 2.7 如果存在 call_type 字段，生成用户标签（用户最常见的 call_type）
if 'call_type' in data.columns:
    logger.info("生成用户级别的 call_type 标签...")
    user_call_type = data.groupby('calling_nbr')['call_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
    user_data = pd.merge(user_data, user_call_type, on='calling_nbr', how='left')
    user_data.rename(columns={'call_type': 'user_call_type'}, inplace=True)
else:
    logger.info("数据集中不存在 call_type 字段，跳过生成用户标签的步骤。")

# 3. 准备聚类特征，不包含用户标识和用户标签列
logger.info("准备聚类特征...")
if 'user_call_type' in user_data.columns:
    cluster_features = [col for col in user_data.columns if col not in ['calling_nbr', 'user_call_type']]
else:
    cluster_features = [col for col in user_data.columns if col != 'calling_nbr']

X = user_data[cluster_features]

# 4. 数据标准化
logger.info("开始数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. K-means 聚类分析
logger.info("开始 K-means 聚类分析...")
kmeans = KMeans(n_clusters=3, random_state=42)  # 根据需要调整簇数
user_data['cluster'] = kmeans.fit_predict(X_scaled)

# 6. 聚类结果可视化

# 6.1 可视化聚类结果：平均通话时长 vs. 总通话数
logger.info("可视化聚类结果：平均通话时长 vs. 总通话数...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_data, x='avg_call_duration', y='total_calls', hue='cluster', palette='viridis')
plt.title('User-level K-means Clustering Results')
plt.xlabel('Average Call Duration')
plt.ylabel('Total Calls')
plt.legend(title='Cluster')
plt.show()

# 6.2 查看每个簇的用户特征平均值
logger.info("查看每个簇的用户特征平均值...")
cluster_means = user_data.groupby('cluster').mean().reset_index()
# print(cluster_means)

# 6.3 使用轮廓系数评估聚类结果
logger.info("计算 Silhouette Score 以评估聚类效果...")
silhouette_avg = silhouette_score(X_scaled, user_data['cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# 6.4 可视化每个簇的特征分布（例如：箱线图）
logger.info("可视化每个簇的特征分布（箱线图）...")
numeric_features = ['avg_call_duration', 'total_calls'] + [col for col in user_data.columns if col.startswith('time_slot_')]
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=user_data, palette='viridis')
    plt.title(f'Cluster vs {feature}')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.show()

# 7. 若需要分类分析（假设要预测 user_call_type）
if 'user_call_type' in user_data.columns:
    logger.info("开始分类分析（决策树）...")
    # 准备分类数据, X_class 不包含用户ID和 cluster 以及标签本身
    X_class = user_data[cluster_features]
    y_class = user_data['user_call_type']

    # 使用决策树进行分类
    logger.info("训练决策树分类模型...")
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_scaled, y_class)

    # 可视化决策树
    logger.info("可视化决策树...")
    plt.figure(figsize=(20, 10))
    class_names = [str(cls) for cls in y_class.unique()]
    plot_tree(dtree, filled=True, feature_names=cluster_features, class_names=class_names, fontsize=10)
    plt.show()

    # 预测与评估
    logger.info("预测并评估分类模型...")
    y_pred = dtree.predict(X_scaled)
    print("Decision Tree Accuracy:", accuracy_score(y_class, y_pred))

    # 交叉验证
    logger.info("进行交叉验证评估分类模型...")
    scores = cross_val_score(dtree, X_scaled, y_class, cv=5)
    print(f'Cross-validation scores: {scores}')
else:
    logger.info("无需进行分类分析，跳过相关步骤。")

# 8. 清理内存
logger.info("清理内存...")
del data, user_features, time_slot_counts, time_slot_props
if 'user_call_type' in locals():
    del user_call_type
gc.collect()

logger.info("代码执行完毕。")
