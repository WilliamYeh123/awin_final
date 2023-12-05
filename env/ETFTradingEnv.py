import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import csv

# 持有資金的上限
MAX_ACCOUNT_BALANCE = 2147483647

# 持有股票數量的上限
MAX_NUM_SHARES = 2147483647

# 股價最高值
MAX_SHARE_PRICE = 5000

# 初始資金
INITIAL_ACCOUNT_BALANCE = 10000

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, df2):
        super(StockTradingEnv, self).__init__()
        
        # 基礎股價資訊
        self.df = df
        
        # 觀測值資訊
        self.df2 = df2
        
        # 獎勵值範圍
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)

        # 只決定是否持有股票，要馬全買(資產竟可能轉換成股票)，要馬全賣(資產全換成現金)
        # 0: all cash
        # 1: all stock
        self.action_space = spaces.Discrete(2)

        # 觀測值Size
        self.observation_space = spaces.Box(
            #low=0, high=1, shape=(220,), dtype=np.float16)
            low=0, high=1, shape=(180,), dtype=np.float16)

    def _next_observation(self):
        obs = self.df2.loc[self.current_step]
        return obs

    def _take_action(self, action):

        # 一般化的化建議取最高最低價之間隨意的價格，但為了比較其他組實驗上的方便，購入價格定為收盤價
        
        #print(self.current_step, len(self.df), self.df.loc[self.current_step, "date"])
        current_price = self.df.loc[self.current_step, "close"]
        
        if action == 0:
            # all cash
            if self.shares_held > 0:
                # 手上有股票，全賣掉
                self.actual_action = 1
                self.sell_at_price = current_price
                self.fee = 0.001425 * self.shares_held * current_price
                self.tax = 0.003 * self.shares_held * current_price
                self.balance += self.shares_held * current_price - self.fee - self.tax
                self.shares_held = 0
                self.net_worth = self.balance + self.shares_held * current_price
            else:
                # 手上沒股票，更新淨利
                self.actual_action = 2
                self.fee = 0
                self.tax = 0
                self.net_worth = self.balance + self.shares_held * current_price

        elif action == 1:
            # all stock
            
            if self.shares_held == 0:
                # 手上沒股票，買進
                self.actual_action = 0
                self.buy_at_price = current_price
                self.shares_held = int(self.balance / current_price)
                self.fee = 0.001425 * self.shares_held * current_price
                self.tax = 0
                self.balance -= self.shares_held * current_price + self.fee
                self.net_worth = self.balance + self.shares_held * current_price
            else:
                # 手上有股票，更新淨利
                self.actual_action = 2
                self.fee = 0
                self.tax = 0
                self.net_worth = self.balance + self.shares_held * current_price
    
    def _get_reward(self):
        
        try:
            previous_price = self.df.loc[self.current_step - 1, "close"]
        except:
            previous_price = self.df.loc[self.current_step, "open"]
        
        current_price = self.df.loc[self.current_step, "close"]
        
        RR = (current_price - previous_price) / previous_price
        
        
        if self.actual_action == 0:
            # buy
            reward = RR - 0.001425
            
        elif self.actual_action == 1:
            # sell
            # 賣的時候就是看收益率/投資報酬率
            RoR = (self.sell_at_price - self.buy_at_price) / self.buy_at_price
            reward = RoR - 0.001425 - 0.003
            
        elif self.actual_action == 2:
            # hold
            # RR正的時候盡量持有，負的時候就不要持有
            reward = RR
            
        return reward
    
    
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        
        if self.current_step > len(self.df.loc[:, 'open'].values) - 2:
            self.current_step = 0
            self.done = True
        else:
            self.done = False
    
        reward = self._get_reward()
        self.current_step += 1
        obs = self._next_observation()

        return obs, reward, False, {}

    def reset(self):
      
        # 口袋還有多少錢
        self.balance = INITIAL_ACCOUNT_BALANCE
        
        # 這一個STEP淨利多少
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        
        # 上一個STEP淨利多少
        self.pre_net_worth = INITIAL_ACCOUNT_BALANCE
            
        # 持有多少張股票
        self.shares_held = 0
        
        # 買進的價格
        self.buy_at_price = 0
        
        # 賣出的價格
        self.sell_at_price = 0
        
        # 實際的動作，0 買 1 賣 2沒動作
        self.actual_action = -1
        
        # 手續費
        self.fee = 0
        
        # 交易稅
        self.tax = 0
        
        self.trades = []
        
        self.current_step = 0
        
        return self._next_observation()
    
    def _render_to_file(self, filename='render.csv'):

        file = open(filename, 'a+', newline='')
        writer = csv.writer(file)
        if self.actual_action == 0:
            #file.write(f'BUY\n')
            writer.writerow([self.current_step, 'buy', self.df.loc[self.current_step-1, "close"]])
        elif self.actual_action == 1:
            writer.writerow([self.current_step, 'sell', self.df.loc[self.current_step-1, "close"]])
        else:
            writer.writerow([self.current_step, 'hold', self.df.loc[self.current_step-1, "close"]])

        file.close()
    
    def render(self, mode='human', close=False, **kwargs):
        
        #print(self.current_step, self.done)
        if self.current_step == 1 and self.done == True:
            self.current_step = len(self.df)
        
        #print(f'Step: {self.current_step}')
        #print(self.df.loc[self.current_step-1, "date"])
        #print("Close at:", self.df.loc[self.current_step-1, "close"])

        if self.actual_action == 0:
            pass
            #print('BUY!! at:', self.df.loc[self.current_step-1, "close"])
            #print('Amount: ', self.shares_held * self.df.loc[self.current_step-1, "close"], "(", self.shares_held, ")")
        if self.actual_action  == 1:
            pass
            #print('SELD!! at:', self.df.loc[self.current_step-1, "close"])
            #print('Amount: ', self.shares_held* self.df.loc[self.current_step-1, "close"], "(", self.shares_held, ")")
        if self.actual_action == 2:
            pass
            #print('HOLD!!!')
        #print()
        #print(f'Fee: {self.fee}')
        #print(f'Tax: {self.tax}')
        #print(f'Balance: {self.balance}')
        #print(f'Shares held: {self.shares_held}')
        #print("Final Assets: ", self.balance + self.shares_held * self.df.loc[self.current_step-1, "close"])
        #print("")
        self._render_to_file(kwargs.get('filename', 'render.csv'))
        #file = open(filename, 'a+')
        #file.write(f'\n')
        #file.close()
        #print('====================================================\n')