import gym
import json
import datetime as dt
import csv

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.sac.policies import MlpPolicy

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import SAC
from stable_baselines import DDPG
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj

#from env.StockTradingEnvIRL import StockTradingEnvIRL
from env.StockTradingEnv import StockTradingEnv
import pandas as pd
from FinMind.data import DataLoader

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

env = DummyVecEnv([lambda: StockTradingEnv(df, df2)])

#model = SAC(MlpPolicy, env, verbose=1)
#model = PPO2('MlpPolicy', env, verbose=1)

############
dataset = ExpertDataset(expert_path='./data/npz/2330_BBAND_trajectory_50.npz', verbose=1, traj_limitation=100, batch_size=128)
model = GAIL('MlpPolicy', env, dataset, verbose=1)
model.learn(total_timesteps=200000)
model.save("./model/Expert_BBAND_2330_200K")

del model # remove to demonstrate saving and loading

model = GAIL.load("./model/Expert_BBAND_2330_200K")
env = DummyVecEnv([lambda: StockTradingEnv(df, df2)])

#model = GAIL.load("2330_PPO2_IRL")
#env = DummyVecEnv([lambda: StockTradingEnv(df, df2)])
obs = env.reset()
for i in range(len(df2)-1):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()