import gym
import json
import datetime as dt
import csv

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.sac.policies import MlpPolicy

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
#from stable_baselines import PPO1
from stable_baselines import A2C
#from stable_baselines import SAC
#from stable_baselines import DDPG
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from env.StockTradingEnv2 import StockTradingEnv


from stable_baselines3 import PPO

import pandas as pd
from FinMind.data import DataLoader

#df = pd.read_csv('./data/2330.TW_Training.csv')
#df = df.sort_values('Date')

api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0xMi0yNyAxNDo1OTowOSIsInVzZXJfaWQiOiJkdXJhbnQ3MTA5MTYiLCJpcCI6IjE0MC4xMjAuMTMuMjMwIn0.8-KIC3-OA4D6JcOtQ_fJBOVkyugx60t1Gy82c57TLz4"

api = DataLoader()
api.login_by_token(api_token = api_token)

# 設定股票標的和開始/結束日期
stock_id = "2330"
start_date='2013-01-01'
end_date='2020-12-31'

# 股價日成交資訊
df = api.taiwan_stock_daily(
    stock_id = stock_id,
    start_date = start_date,
    end_date = end_date
)

print(len(df))

df2 = pd.read_csv('./data/Input/2330_input.csv',header=None)

print(len(df2))

#print(df2.head())
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df, df2)])

model = PPO2('MlpPolicy', env, verbose=1)
#model = PPO2('MlpPolicy', env, verbose=1)
#model = PPO("MlpPolicy", env, verbose=1)

#generate_expert_traj(model, 'expert_william_2', n_timesteps=20000, n_episodes=1) #######20210116, 20210301
#model = A2C(MlpPolicy, env, verbose=1)
#model = DDPG('MlpPolicy', env)

#model.learn(total_timesteps=100000)
#model.save("2330_ppo2")
generate_expert_traj(model, './data/npz/Expert_template', n_timesteps=200000, n_episodes=1)

#model = PPO2.load("2330_ppo2")

obs = env.reset()
for i in range(len(df)-1):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
