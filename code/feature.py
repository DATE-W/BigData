import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 1. 从 data.txt 文件读取数据
df = pd.read_csv('data.txt', sep='\t', header=None,
                   dtype={'calling_nbr': str,
                          'called_nbr': str,
                          'calling_city': str,
                          'called_city': str,
                          'calling_roam_city': str,
                          'called_roam_city': str,
                          'calling_cell': str},  # 强制将这些列作为字符串类型
                   names=['day_id', 'calling_nbr', 'called_nbr', 'calling_optr', 'called_optr',
                          'calling_city', 'called_city', 'calling_roam_city', 'called_roam_city',
                          'start_time', 'end_time', 'raw_dur', 'call_type', 'calling_cell'])

# 2. 解析时间字段
df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time

# 3. 转换通话时长为整数
df['raw_dur'] = df['raw_dur'].astype(int)

# 4. 定义时间段区间函数
def get_time_slot(time_obj):
    """返回对应时间段的编号"""
    if 0 <= time_obj.hour < 3:
        return 'TimeSlot1'
    elif 3 <= time_obj.hour < 6:
        return 'TimeSlot2'
    elif 6 <= time_obj.hour < 9:
        return 'TimeSlot3'
    elif 9 <= time_obj.hour < 12:
        return 'TimeSlot4'
    elif 12 <= time_obj.hour < 15:
        return 'TimeSlot5'
    elif 15 <= time_obj.hour < 18:
        return 'TimeSlot6'
    elif 18 <= time_obj.hour < 21:
        return 'TimeSlot7'
    else:
        return 'TimeSlot8'

# 5. 为每个通话记录生成时间段
df['time_slot'] = df['start_time'].apply(lambda x: get_time_slot(x))

# 6. 先计算所有外呼和接听次数、时间段统计和平均通话时长
# 使用 groupby 来优化
calling_counts = df.groupby('calling_nbr').size()
called_counts = df.groupby('called_nbr').size()

time_slot_counts = df.groupby(['calling_nbr', 'time_slot']).size().unstack(fill_value=0)
duration_avg = df.groupby('calling_nbr')['raw_dur'].mean()

# 7. 统计每个号码的各种指标
result = []

# 获取所有号码列表（主叫号码和被叫号码的并集）
user_ids = pd.concat([df['calling_nbr'], df['called_nbr']]).unique()

# 使用 tqdm 显示进度条
for user_id in tqdm(user_ids, desc="Processing Users", unit="user", ncols=100):
    user_data = {}
    user_data['UserID'] = user_id

    # 外呼和接听的统计
    user_data['OutgoingCalls'] = calling_counts.get(user_id, 0)  # 外呼次数
    user_data['IncomingCalls'] = called_counts.get(user_id, 0)   # 接听次数

    # 获取该用户的时间段统计
    time_slot_data = time_slot_counts.loc[user_id] if user_id in time_slot_counts.index else [0] * 8
    user_data.update(dict(zip(time_slot_counts.columns, time_slot_data)))

    # 获取该用户的平均通话时长
    user_data['AvgDuration'] = duration_avg.get(user_id, np.nan)

    result.append(user_data)

# 8. 转换结果为DataFrame
final_df = pd.DataFrame(result)

# 9. 将结果保存到 clean.txt 文件
final_df.to_csv('clean1.txt', sep='\t', index=False)

print("清洗后的数据已保存到 clean.txt")
