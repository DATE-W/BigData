import pandas as pd

# 读取数据时，确保 calling_nbr 和 called_nbr 作为字符串读取
data = pd.read_csv('data.txt', sep='\t', header=None,
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

# 1. 删除缺失数据行（排除 calling_cell 字段）
non_nullable_columns = ['day_id', 'calling_nbr', 'called_nbr', 'calling_optr', 'called_optr',
                        'calling_city', 'called_city', 'calling_roam_city', 'called_roam_city',
                        'start_time', 'end_time', 'raw_dur', 'call_type']
data = data.dropna(subset=non_nullable_columns)

# 2. 清洗日期和时间字段
# 将start_time和end_time转换为datetime格式
data['start_time'] = pd.to_datetime(data['start_time'], format='%H:%M:%S', errors='coerce').dt.time
data['end_time'] = pd.to_datetime(data['end_time'], format='%H:%M:%S', errors='coerce').dt.time

# 3. 检查raw_dur字段，确保通话时长为正整数
data = data[data['raw_dur'] > 0]  # 保留通话时长大于0的记录

# 4. 检查call_type字段，确保是1、2、3
valid_call_types = [1, 2, 3]
data = data[data['call_type'].isin(valid_call_types)]

# 5. 确保 calling_nbr 和 called_nbr 保持为字符串类型（包括数字前导零）
# 不需要再做转换，因为我们已经在读取时强制它们为字符串

# 6. 清洗完毕后，查看清洗后的数据
print(data.head())

# 7. 将清洗后的数据保存为新的文件（确保以字符串格式保存所有数据）
data.to_csv('cleaned_data.csv', index=False, sep='\t')  # quoting=1 用于确保文本数据（如电话号码）被正确保存

# --- 任务 1：计算每日平均通话次数 ---

# 1. 按 calling_nbr 和 day_id 计算每日的通话次数
daily_calls = data.groupby(['calling_nbr', 'day_id']).size().reset_index(name='call_count')

# 2. 计算每个主叫号码的每日平均通话次数
avg_daily_calls = daily_calls.groupby('calling_nbr')['call_count'].mean().reset_index()
avg_daily_calls.columns = ['calling_nbr', 'avg_daily_calls']

# 保存每日平均通话次数
avg_daily_calls.to_excel('avg_daily_calls.xlsx', index=False)

# --- 任务 2：计算时间段通话时长所占比例 ---

# 1. 定义时间段划分函数
def get_time_slot(hour):
    """根据小时划分时间段"""
    if 0 <= hour < 3:
        return 1  # 时间段1
    elif 3 <= hour < 6:
        return 2  # 时间段2
    elif 6 <= hour < 9:
        return 3  # 时间段3
    elif 9 <= hour < 12:
        return 4  # 时间段4
    elif 12 <= hour < 15:
        return 5  # 时间段5
    elif 15 <= hour < 18:
        return 6  # 时间段6
    elif 18 <= hour < 21:
        return 7  # 时间段7
    else:
        return 8  # 时间段8

# 2. 提取start_time的小时部分，并根据小时划分时间段
data['hour'] = pd.to_datetime(data['start_time'].astype(str), format='%H:%M:%S').dt.hour
data['time_slot'] = data['hour'].apply(get_time_slot)

# 3. 按 calling_nbr 和 time_slot 计算每个时间段的通话时长
time_slot_durations = data.groupby(['calling_nbr', 'time_slot'])['raw_dur'].sum().reset_index()

# 4. 计算每个主叫号码的总通话时长
total_durations = data.groupby('calling_nbr')['raw_dur'].sum().reset_index()
total_durations.columns = ['calling_nbr', 'total_dur']

# 5. 合并数据，计算每个时间段的通话时长占比
result = pd.merge(time_slot_durations, total_durations, on='calling_nbr')
result['duration_ratio'] = result['raw_dur'] / result['total_dur']

# 6. 转换为宽格式：每个时间段一列
final_result = result.pivot(index='calling_nbr', columns='time_slot', values='duration_ratio').reset_index()

# 保存时间段通话时长占比
final_result.to_excel('time_slot_ratios.xlsx', index=False)

# --- 输出示例 ---
# 打印输出查看结果
print("每日平均通话次数：")
print(avg_daily_calls.head())

print("\n时间段占比：")
print(final_result.head())
