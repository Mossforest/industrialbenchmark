"""
The MIT License (MIT)

Copyright 2020 Siemens AG

Authors: Phillip Swazinna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import time

from stable_baselines3.ddpg.policies import MlpPolicy as ddpgPolicy
from stable_baselines3.sac.policies import MlpPolicy as sacPolicy
from stable_baselines3.td3.policies import MlpPolicy as td3Policy
from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3 #, TRPO
from stable_baselines3.common.utils import get_device

from industrial_benchmark_python.IBGym import IBGym

# algorithms = [("a2c", A2C),("ppo", PPO), ("sac", SAC), ("td3", TD3), ("ddpg", DDPG)] #  ("trpo", TRPO),
algorithms = [("ddpg", DDPG)] #  ("trpo", TRPO),
GAMMA = 0.97
N_trials = 5

print("Training the following Algorithms on the industrial benchmark", N_trials, "times each:\n",
      [x[0] for x in algorithms], "\n")

for i in range(N_trials):
    print("Starting training round", i)
    for name, algo in algorithms:
        print("-->", name)
        t1 = time.time()
        # create environment
        env = IBGym(setpoint=70, reward_type='classic', action_type='continuous', observation_type='include_past')

        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        param_noise = None

        if name == "ddpg":
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            A = algo(ddpgPolicy, env, verbose=1, action_noise=action_noise, gamma=GAMMA)
            A.learn(total_timesteps=500000)
        elif name == "td3":
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
            A = algo(td3Policy, env, verbose=1, action_noise=action_noise, gamma=GAMMA)
            A.learn(total_timesteps=50000)
        elif name == "sac":
            A = algo(sacPolicy, env, verbose=1, gamma=GAMMA)
            A.learn(total_timesteps=25000)
        else:
            A = algo(MlpPolicy, env, verbose=1, gamma=GAMMA)
            A.learn(total_timesteps=25000)

        # saving
        A.save("{0}-{1}_IB".format(name, i))
        del A
        print(f'-----> finished {name} in {(time.time() - t1):.2f} sec.')


# evaluating
print("\nDone with trainings, moving on to evaluation\n")
avg_rewards = []
for idx in range(N_trials):
    print("Starting Evaluation round", idx)
    for name, algo in algorithms:
        env = IBGym(setpoint=70, reward_type='classic', action_type='continuous', observation_type='include_past')

        A = algo.load("{0}-{1}_IB".format(name, idx))
        num_trajectories = 100
        num_steps = 100

        obs = env.reset()
        rewards = []
        for i in range(num_trajectories):
            for j in range(num_steps):
                action, _states = A.predict(obs)
                obs, reward, done, info = env.step(action)
                rewards.append(reward * GAMMA**j)
        avg_rewards.append((name, idx, sum(rewards) / float(num_trajectories)))

print("\nFinal Results per Algorithm averaged over the", N_trials, "rounds:\n", avg_rewards)
