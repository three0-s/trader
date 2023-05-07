import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env import CryptoMarketEnv
import pandas as pd

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: CryptoMarketEnv('/Users/yewon/Documents/traderWon/envs/data', 14, 0.2, 0.7, '/Users/yewon/Documents/traderWon/envs/test_render')])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000)
obs = env.reset()
for i in range(120):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()