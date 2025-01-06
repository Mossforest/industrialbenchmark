from industrial_benchmark_python.IBGym import IBGym
import numpy as np
import pickle
import time

DISCOUNT = 0.97
T = 100
train_n_trajectories = 1_000
test_n_trajectories = 100
setpoint=[70]# , 20, 50] #70

def collect_multiplesample(setp, n_trajectories, eval, n_samples=5):
    env = IBGym(setpoint=setp, reward_type='classic', action_type='continuous', observation_type='classic')
    data = {'s': [], 'a': [], 'r': [], 'next_s_samples': [], 'r_samples': []}
    returns = []
    t1 = time.time()
    
    for k in range(n_trajectories):
        traj = {'s': [], 'a': [], 'r': [], 'next_s_samples': [], 'r_samples': []}
        acc_return = 0.
        
        s0 = env.reset()
        traj['s'].append(s0)
        traj['a'].append(np.zeros(3))
        traj['r'].append(0)
        traj['next_s_samples'].append([s0] * n_samples)  # 初始状态的多个样本
        traj['r_samples'].append([0] * n_samples)
        
        for i in range(T):
            action = env.action_space.sample()
            # 获取多个样本
            results = env.step(action, n_samples=n_samples)
            
            # 分离多个样本的状态和奖励
            next_states = [r[0] for r in results]
            rewards = [r[1] for r in results]
            
            # 使用最后一个样本更新环境状态
            state = next_states[-1]
            reward = rewards[-1]
            
            acc_return += reward * DISCOUNT**i
            
            # 保存当前状态、动作和奖励
            traj['s'].append(state)
            traj['a'].append(action)
            traj['r'].append(reward)
            
            # 保存所有样本
            traj['next_s_samples'].append(next_states)
            traj['r_samples'].append(rewards)
        
        returns.append(acc_return / T)
        data['s'].append(np.stack(traj['s'], axis=0))
        data['a'].append(np.stack(traj['a'], axis=0))
        data['r'].append(np.stack(traj['r'], axis=0))
        data['next_s_samples'].append(np.array(traj['next_s_samples']))
        data['r_samples'].append(np.array(traj['r_samples']))

        if k % 10 == 0:
            print(f'data: {len(data["s"])}')
            print("random actions achieved return", np.mean(returns), "+-", np.std(returns), "time", time.time() - t1)
            t1 = time.time()

    data['s'] = np.stack(data['s'], axis=0)
    data['a'] = np.stack(data['a'], axis=0)
    data['r'] = np.stack(data['r'], axis=0)
    data['next_s_samples'] = np.stack(data['next_s_samples'], axis=0)
    data['r_samples'] = np.stack(data['r_samples'], axis=0)
    
    # 保存数据
    file_name = f'./data/data_{setp}_multisample.pkl' if not eval else f'./data/data_{setp}_multisample_eval.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

for setp in setpoint:
    collect_multiplesample(setp, train_n_trajectories, False)
    collect_multiplesample(setp, test_n_trajectories, True)