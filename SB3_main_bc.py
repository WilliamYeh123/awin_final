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
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

import gym
import seals
import os


api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0xMi0yNyAxNDo1OTowOSIsInVzZXJfaWQiOiJkdXJhbnQ3MTA5MTYiLCJpcCI6IjE0MC4xMjAuMTMuMjMwIn0.8-KIC3-OA4D6JcOtQ_fJBOVkyugx60t1Gy82c57TLz4"

api = DataLoader()
api.login_by_token(api_token = api_token)

# 設定股票標的和開始/結束日期
stock_list = ['2330', '2603', '2002','1301', '2801']
strategy_list = ['SMA', 'KD', 'BBAND']

stock_id = "2330"
strategy = "SMA"
day_length = 20
flag = 2

#start_date='2001-01-01'
#start_date='2013-01-01'
start_date = '2021-01-01'
end_date = '2021-12-31'
#end_date='2020-12-31'

for stock_id in stock_list:
    for strategy in strategy_list:
        print(stock_id, strategy, day_length)

        # 股價日成交資訊
        df = api.taiwan_stock_daily(
            stock_id = stock_id,
            start_date = start_date,
            end_date = end_date
        )

        #df2 = pd.read_csv('./data/Input/' + stock_id + '_input_train_' + str(day_length) + "_NEW2" + '.csv',header=None)
        #df2 = pd.read_csv('./data/Input/' + stock_id + '_input_train_' + str(day_length) + "_NEW" + '.csv',header=None)
        #df2 = pd.read_csv('./data/Input/temp/' + stock_id + '_input_train_' + str(day_length) + '.csv',header=None)

        df2 = pd.read_csv('./data/Input/' + stock_id + '_input_test_20_NEW2.csv',header=None)

        print(len(df))
        print(len(df2))

        venv = DummyVecEnv([lambda: StockTradingEnv(df, df2)])

        expert = PPO(
            policy=MlpPolicy,
            env=venv,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
        )

        trajectory_list = []

        #filename = stock_id + '_' + strategy + '_trajectory_50_train.csv'
        filename = stock_id + '_' + strategy + '_trajectory_50_test.csv'
        #filename = stock_id + '_' + strategy + '_trajectory_50.csv'

        #filename = '2330_ZIGZAG_trajectory_0.02.csv'

        #"""
        with open("./data/Trajectory/Test/" + filename, 'r', encoding='utf8', newline='') as csvFile:
            reader = csv.reader(csvFile)
            for r in reader:
                trajectory_list.append(r)
        #"""

        """
        with open("./data/Trajectory/Test/" + filename, 'r', encoding='utf8', newline='') as csvFile:
            reader = csv.reader(csvFile)
            for r in reader:
                trajectory_list.append(r)
        """



        final_trajectory = []
        state = 0

        # 0: all cash
        # 1: all stock

        for trajectory in trajectory_list:
            if trajectory[1] == 'buy':
                final_trajectory.append(1)
                state = 1
            elif trajectory[1] == 'sell':
                final_trajectory.append(0)
                state = 0
            else:
                if state == 0:
                    final_trajectory.append(0)
                else:
                    final_trajectory.append(1)

        shares_held = 0
        reward_list = []

        for i in range(len(final_trajectory)):
            price = df.loc[i, "close"]
            try:
                previous_price = df.loc[i - 1, "close"]
            except:
                previous_price = df.loc[i, "open"]

            if final_trajectory[i] == 0:
                if shares_held > 0:
                    sell_at_price = price
                    shares_held = 0
                    actual_action = 1
                else:
                    actual_action = 2
            elif final_trajectory[i] == 1:
                if shares_held == 0:
                    buy_at_price = price
                    shares_held = 1
                    actual_action = 0
                else:
                    actual_action = 2

            current_price = df.loc[i, "close"]


            RR = (current_price - previous_price) / previous_price

            if actual_action == 0:
                # buy
                reward = RR

            elif actual_action == 1:
                # sell
                # 賣的時候就是看收益率/投資報酬率
                RoR = (sell_at_price - buy_at_price) / buy_at_price
                reward = RoR

            elif actual_action == 2:
                # hold
                # RR正的時候盡量持有，負的時候就不要持有
                reward = RR

            reward_list.append(reward)

        t_action = np.array(final_trajectory)
        t_reward = np.array(reward_list)

        trajectory = types.TrajectoryWithRew(obs=df2, acts=t_action, infos=None, terminal=True, rews=t_reward)


        venv = DummyVecEnv([lambda: StockTradingEnv(df, df2)])

        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=[trajectory],
        )

        if flag == 1:
            model = bc.reconstruct_policy(policy_path="./model/BC_" + strategy + "_" + stock_id + "_" + str(day_length) + "_train_200")
        else:
            model = bc.reconstruct_policy(policy_path="./model/BC_" + strategy + "_" + stock_id + "_" + str(day_length) + "_train")

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

        if stock_value != 0:
            temp.append(float(trajectory[2]))
            r = (temp[1]-temp[0])/temp[0]
            temp.append(r)
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

        if flag == 1:
            os.renames("./render.csv", "./result/BC/" + stock_id + "_" + strategy + "_200" + ".csv")
        else:
            os.renames("./render.csv", "./result/BC/" + stock_id + "_" + strategy + "_2000" + ".csv")