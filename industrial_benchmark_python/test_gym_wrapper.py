from industrial_benchmark_python.IBGym import IBGym
import numpy as np
import pickle
import time

DISCOUNT = 0.97
T = 100
train_n_trajectories = 10_000
test_n_trajectories = 1_000
setpoint=[20, 50] #70

def collect(setp, n_trajectories, eval):
    env = IBGym(setpoint=setp, reward_type='classic', action_type='continuous', observation_type='classic')
    data = {'s': [], 'a': [], 'r': []}
    returns = []
    t1 = time.time()
    for k in range(n_trajectories):
        traj = {'s': [], 'a': [], 'r': []}
        acc_return = 0.
        s0 = env.reset()
        traj['s'].append(s0)
        traj['a'].append(np.zeros(3))
        traj['r'].append(0)
        for i in range(T):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            acc_return += reward * DISCOUNT**i
            traj['s'].append(state)
            traj['a'].append(action)
            traj['r'].append(reward)
        returns.append(acc_return / T)
        data['s'].append(np.stack(traj['s'], axis=0))
        data['a'].append(np.stack(traj['a'], axis=0))
        data['r'].append(np.stack(traj['r'], axis=0))

        if k % 1000 == 0:
            print(f'data: {len(data["s"])}')
            print("random actions achieved return", np.mean(returns), "+-", np.std(returns), "time", time.time() - t1)
            t1 = time.time()

    data['s'] = np.stack(data['s'], axis=0)
    data['a'] = np.stack(data['a'], axis=0)
    data['r'] = np.stack(data['r'], axis=0)
    file_name = f'./data/data_{setp}.pkl' if not eval else f'./data/data_{setp}_eval.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

for setp in setpoint:
    collect(setp, train_n_trajectories, False)
    collect(setp, test_n_trajectories, True)