from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
from env.StockTradingEnv2 import StockTradingEnv
import pandas as pd
from FinMind.data import DataLoader
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import csv
import numpy as np
from imitation.data import types
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import gym
import seals
import os


api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0xMi0yNyAxNDo1OTowOSIsInVzZXJfaWQiOiJkdXJhbnQ3MTA5MTYiLCJpcCI6IjE0MC4xMjAuMTMuMjMwIn0.8-KIC3-OA4D6JcOtQ_fJBOVkyugx60t1Gy82c57TLz4"

api = DataLoader()
api.login_by_token(api_token = api_token)

# 設定股票標的和開始/結束日期

stock_list = ['2330', '2603', '2002','1301', '2801']
strategy_list = ['SMA', 'KD', 'BBAND']

stock_list = ['2603']
strategy_list = ['BBAND']

#stock_id = '2330'
#strategy = "BBAND"
#start_date='2013-01-01'

start_date='2001-01-01'
end_date='2020-12-31'

#start_date='2021-01-01'
#end_date='2021-12-31'


#df = pd.read_csv('./data/' + stock_id + ".csv")
#df = pd.read_csv('./data/' + stock_id + "_test.csv")

# 股價日成交資訊
"""
df = api.taiwan_stock_daily(
    stock_id = stock_id,
    start_date = start_date,
    end_date = end_date
)
"""


#20
#df2 = pd.read_csv('./data/Input/' + stock_id + '_input_train_20_NEW.csv',header=None)

#10
#df2 = pd.read_csv('./data/Input/' + stock_id + '_input_train_10_NEW2.csv',header=None)

#nochip
#df2 = pd.read_csv('./data/Input/' + stock_id + '_input_train_20_nochip.csv',header=None)

#df2 = pd.read_csv('./data/Input/' + stock_id + '_input_test_20_NEW2.csv',header=None)

#print(len(df), len(df2))

#venv = DummyVecEnv([lambda: StockTradingEnv(df, df2)])


for stock_id in stock_list:
    df = pd.read_csv('./data/' + stock_id + ".csv")
    df2 = pd.read_csv('./data/Input/' + stock_id + '_input_train_20_NEW.csv',header=None)
    venv = DummyVecEnv([lambda: StockTradingEnv(df, df2)])
    
    
    for strategy in strategy_list:    
        
        
        
        
        
        
        #20
        model = PPO.load("./model/Expert_" + strategy + "_" + stock_id + "_20_2M_NEW")
        
        #10
        #model = PPO.load("./model/Expert_" + strategy + "_" + stock_id + "_10_train_NEW")
        
        #nochip
        #model = PPO.load("./model/Expert_" + strategy + "_" + stock_id + "_20_train_nochip")
        #model = PPO.load("./model/Expert_" + strategy + "_" + stock_id + "_20_train_nochip_2")

        #model = PPO.load("./model/Expert_" + strategy + "_" + stock_id + "_20_final_10")


        for count in range(4, 5):

            obs = venv.reset()
            for i in range(len(df)):
                action, _states = model.predict(obs)
                obs, rewards, done, info = venv.step(action)
                venv.render()

            fee = 0.001425
            tax = 0.003
            with open("render.csv", 'r', encoding = 'utf8', newline = '') as csvFile:
                reader = csv.reader(csvFile)
                trajectory_list = [r for r in reader]

            MAX_ACCOUNT_BALANCE = 10000

            balance = MAX_ACCOUNT_BALANCE
            net_worth = MAX_ACCOUNT_BALANCE
            stock_num = 0
            stock_value = 0

            buy_sell_tuple = []
            temp = []

            for trajectory in trajectory_list[:]:
                if trajectory[1] == 'buy':
                    stock_num = int(balance / float(trajectory[2]))
                    stock_value = stock_num * float(trajectory[2])
                    balance = balance - stock_value - stock_value * fee
                    #print("Buy at", trajectory[2])
                    #print(balance+stock_value)
                    #print()

                    temp.append(float(trajectory[2]))

                elif trajectory[1] == 'sell':
                    stock_value = stock_num * float(trajectory[2])
                    balance = balance + stock_value - stock_value * (fee + tax)
                    stock_num = 0
                    stock_value = 0
                    #print("Sell at", trajectory[2])
                    #print(balance+stock_value)
                    #print()

                    temp.append(float(trajectory[2]))
                    r = (temp[1]-temp[0])/temp[0]
                    temp.append(r)
                    buy_sell_tuple.append(temp)
                    temp = []
                elif trajectory[1] == 'hold':
                    stock_value = stock_num * float(trajectory[2])
                    #print(balance + stock_value)
                    #print()

            if len(temp) != 0:
                temp.append(float(trajectory[2]))
                r = (temp[1]-temp[0])/temp[0]
                temp.append(r)
                if len(temp) != 3:
                    print("Wrong")
                buy_sell_tuple.append(temp)
                temp = []

            total_num = len(buy_sell_tuple)
            total_sum = 0
            for t in buy_sell_tuple:
                total_sum += t[2] 


            print(stock_id, strategy)
            print('平均報酬率：', total_sum/total_num)
            print('交易次數：',  total_num)
            print('最終收益', balance+stock_value)

            os.renames("./render.csv", "./result/GAIL/Train/nochips/" + stock_id + "_" + strategy + "_" + str(count) + ".csv")

        print()

    
    
    
