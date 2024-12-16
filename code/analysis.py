import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
data = pd.read_csv('cleaned_data.csv', sep='\t', header=0,
                   dtype={'calling_nbr': str,
                          'called_nbr': str,
                          'calling_city': str,
                          'called_city': str,
                          'calling_roam_city': str,
                          'called_roam_city': str,
                          'calling_cell': str})

# 2. 数据预处理：特征提取
# 用户是否为主叫（主叫 = 1， 被叫 = 0）
data['is_caller'] = np.where(data['calling_nbr'] == data['called_nbr'], 0, 1)

# 将通话开始时间转换为小时（时间段）
data['start_hour'] = pd.to_datetime(data['start_time'].astype(str), format='%H:%M:%S').dt.hour
data['time_slot'] = data['start_hour'] // 3 + 1  # 将时间划分为8个时间段（0-3, 3-6, ..., 21-24）

# 计算通话时长
data['call_duration'] = data['raw_dur']

# 提取其他行为特征，如每个用户的平均通话时长
data['avg_call_duration'] = data.groupby('calling_nbr')['raw_dur'].transform('mean')

# 每个用户的通话次数
data['total_calls'] = data.groupby('calling_nbr')['raw_dur'].transform('count')

# 3. 数据标准化（对于K-means和决策树来说，标准化是必要的）
features = ['call_duration', 'avg_call_duration', 'total_calls', 'is_caller', 'time_slot']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 选择聚类或分类模型所需的特征
X = data[['call_duration', 'avg_call_duration', 'total_calls', 'is_caller', 'time_slot']]

# 4. K-means 聚类分析
kmeans = KMeans(n_clusters=3, random_state=42)  # 假设分为3个簇
data['cluster'] = kmeans.fit_predict(X)

# 5. 聚类结果可视化：展示不同簇的用户群体
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='call_duration', y='avg_call_duration', hue='cluster', palette='viridis')
plt.title('K-means Clustering Results')
plt.xlabel('Call Duration')
plt.ylabel('Average Call Duration')
plt.legend(title='Cluster')
plt.show()

# 查看每个簇的用户特征
# print(data.groupby('cluster').mean())

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 6. 分类分析：决策树
X_class = data[['call_duration', 'avg_call_duration', 'total_calls', 'is_caller', 'time_slot']]
y_class = data['call_type']  # 假设我们要预测通话类型

# 使用决策树进行分类
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_class, y_class)

# 7. 可视化决策树
# 确保 class_names 是一个列表，并且每个类别为字符串
class_names = [str(cls) for cls in y_class.unique()]  # 将类别转换为字符串并创建列表

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(dtree, filled=True, feature_names=X_class.columns, class_names=class_names, fontsize=10)
plt.show()

# 8. 分类结果预测
y_pred = dtree.predict(X_class)


# 评估分类效果：准确性
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_class, y_pred))

# 可视化聚类结果（通话时长 vs. 平均通话时长）
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='call_duration', y='avg_call_duration', hue='cluster', palette='viridis')
plt.title('K-means Clustering Results')
plt.xlabel('Call Duration')
plt.ylabel('Average Call Duration')
plt.legend(title='Cluster')
plt.show()

# 决策树分类可视化（已经在上面用 plot_tree 展示过）

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

from sklearn.model_selection import cross_val_score
scores = cross_val_score(dtree, X_class, y_class, cv=5)
print(f'Cross-validation scores: {scores}')
