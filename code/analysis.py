import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 从 clean.txt 文件读取数据
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', usecols=range(1, len(pd.read_csv(file_path, sep='\t', nrows=1).columns)))
    return df

# 3. 归一化特征
def normalize_features(df, features):
    scaler = StandardScaler()
    # 删除包含缺失值的行
    df = df.dropna(subset=features)  # 去掉这些包含NaN的行
    df[features] = scaler.fit_transform(df[features])
    return df


# 4. 使用肘部法则（Elbow Method）确定聚类数
def find_optimal_k(df, features):
    sse = []
    k_range = range(1, 11)  # 测试从 1 到 10 个簇
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        kmeans.fit(df[features])
        sse.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.show()

    # 返回肘部法则最佳 K 值
    optimal_k = k_range[np.argmin(np.diff(sse)) + 1]  # 找到最佳的 K 值
    return optimal_k


# 5. 聚类分析：使用 KMeans 聚类
def perform_clustering(df, features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features])
    return df, kmeans


# 6. 可视化：聚类结果的散点图（基于 OutgoingCalls 和 IncomingCalls）
def plot_cluster_scatter(df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='OutgoingCalls', y='IncomingCalls', hue='Cluster', palette='viridis', s=100)
    plt.title('Clustering of Users Based on Outgoing and Incoming Calls', fontsize=16)
    plt.xlabel('Outgoing Calls', fontsize=14)
    plt.ylabel('Incoming Calls', fontsize=14)
    plt.legend(title='Cluster', fontsize=12)
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.show()


# 7. 标注每个类别的行为特征
def label_clusters(df):
    # 聚类结果的特征均值
    cluster_summary = df.groupby('Cluster').mean()

    print("\n每个聚类类别的行为特征：")
    for i in range(len(cluster_summary)):
        print(f"\nCluster {i} behavior:")
        for feature, value in cluster_summary.iloc[i].items():
            print(f"{feature}: {value:.4f}")


# 8. 绘制雷达图
def plot_radar(df, features, title, save_path=None):
    # 计算每个聚类类别的平均值
    cluster_summary = df.groupby('Cluster').mean()

    categories = features  # 用于雷达图的特征类别
    for i in range(len(cluster_summary)):
        values = cluster_summary.iloc[i].values
        values = np.concatenate((values, values[:1]))  # 为了形成闭合图形
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        plt.subplot(111, polar=True)
        plt.fill(angles, values, color='blue', alpha=0.25)
        plt.plot(angles, values, color='blue', linewidth=2)
        plt.title(f'{title} - Cluster {i}')
        plt.xticks(angles[:-1], categories, color='black', size=10)
        if save_path:
            plt.savefig(f"{save_path}_cluster_{i}.png", format='png', bbox_inches='tight')
        plt.show()


# 9. 保存结果
def save_results(df, output_file):
    df.to_csv(output_file, sep='\t', index=False)
    print(f"聚类结果已保存至 {output_file}")


# 主函数
def main():
    # 数据文件路径
    file_path = 'clean.txt'

    # 选择需要归一化的特征
    features_to_normalize = [
        'OutgoingCalls', 'IncomingCalls', 'TimeSlot1', 'TimeSlot2',
        'TimeSlot3', 'TimeSlot4', 'TimeSlot5', 'TimeSlot6', 'TimeSlot7', 'TimeSlot8',
        'AvgDuration'
    ]

    # 读取数据
    df = load_data(file_path)
    print(df.head())  # 打印数据前几行，确保数据读取正确

    # 归一化特征
    df = normalize_features(df, features_to_normalize)
    print(df.head())

    # 使用肘部法则选择合适的簇数
    optimal_k = find_optimal_k(df, features_to_normalize)

    # 根据肘部法则选择的最优簇数进行聚类
    df, kmeans = perform_clustering(df, features_to_normalize, n_clusters=4)

    # 可视化基于 OutgoingCalls 和 IncomingCalls 的聚类散点图
    plot_cluster_scatter(df, save_path='cluster_scatter_plot.png')

    # 绘制每个聚类类别的雷达图
    plot_radar(df, features_to_normalize, title='Cluster Behavior Analysis', save_path='cluster_radar')

    # 标注每个类别的行为特征
    label_clusters(df)

    # 保存聚类结果到新文件
    save_results(df, 'clustered_clean_data.txt')


if __name__ == '__main__':
    main()
