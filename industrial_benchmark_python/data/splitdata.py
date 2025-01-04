import pickle
import random
import numpy as np

# 替换为你的.pkl文件路径
file_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/industrialbenchmark-master/industrial_benchmark_python/data/data_demo.pkl'
save_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/industrialbenchmark-master/industrial_benchmark_python/data/data_demo_sampled100.pkl'

# 使用pickle.load()函数加载.pkl文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 查看数据类型
print(type(data))

# 如果数据是字典或列表，可以查看其键或元素
if isinstance(data, dict):
    print(data.keys())
elif isinstance(data, list):
    print(data[0].keys())  # 假设列表中的元素是字典

random.seed(212)
np.random.seed(212)
indexes = random.sample(range(len(data['s'])), 100)
indexes = np.array(indexes, dtype=int)

sampled_data = {}
for k in data.keys():
    sampled_data[k] = data[k][indexes]
    print(f'{k}: {sampled_data[k].shape}, {type(sampled_data[k])}')

with open(save_path, 'wb') as file:
    pickle.dump(sampled_data, file)