import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

# 持有資金的上限
MAX_ACCOUNT_BALANCE = 2147483647

# 持有股票數量的上限
MAX_NUM_SHARES = 2147483647

# 股價最高值
MAX_SHARE_PRICE = 5000

# 初始資金
INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    visualization = None
    
    def __init__(self, df, df2):
        super(StockTradingEnv, self).__init__()
        
        # 基礎股價資訊
        self.df = df
        
        # 觀測值資訊
        self.df2 = df2
        
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # 只決定是否持有股票，要馬全買(資產竟可能轉換成股票)，要馬全賣(資產全換成現金)
        # 0: all cash
        # 1: all stock
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(220,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(180,), dtype=np.float32)


    def _next_observation(self):        
        obs = self.df2.loc[self.current_step]     
        return obs

    def _take_action(self, action):
        
        # 購入價格為收盤價
        price = self.df.loc[self.current_step, "close"]
        
        self.pre_net_worth = self.net_worth * 1

        action_type = action
        
        amount = 1
        
        if action == 0:
            # all cash
            if self.shares_held > 0:
                # 手上有股票，全賣掉
                #print("手上有股票，全賣掉")
                self.actual_action = 1
                self.sell_at_price = price
                self.fee = 0.001425 * self.shares_held * price
                self.tax = 0.003 * self.shares_held * price
                self.balance += self.shares_held * price - self.fee - self.tax
                self.shares_held = 0
                self.net_worth = self.balance + self.shares_held * price
            else:
                # 手上沒股票，更新淨利
                #print("手上沒股票，更新淨利")
                self.actual_action = 2
                self.fee = 0
                self.tax = 0
                self.net_worth = self.balance + self.shares_held * price

        elif action == 1:
            # all stock
            
            if self.shares_held == 0:
                # 手上沒股票，買進
                #print("手上沒股票，買進")
                self.actual_action = 0
                self.buy_at_price = price
                self.shares_held = int(self.balance / price)
                self.fee = 0.001425 * self.shares_held * price
                self.tax = 0
                self.balance -= self.shares_held * price + self.fee
                self.net_worth = self.balance + self.shares_held * price
            else:
                # 手上有股票，更新淨利
                #print("手上有股票，更新淨利")
                self.actual_action = 2
                self.fee = 0
                self.tax = 0
                self.net_worth = self.balance + self.shares_held * price

        else:
            print("Action space is wrong.")
        
       
            
    def _get_reward(self):
        #print(self.current_step, "get reward")
        
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
        #print(self.current_step, self.df.loc[self.current_step, "date"])
                
        if self.df.loc[self.current_step, "date"] == '2020-12-31':
            print('Final Step')
            print(action)
            self._take_action(action)
            #self.current_step += 1
            done = True
            reward = 0
            obs = self._next_observation()
            self.current_step += 1
        else:
            self._take_action(action)
            done = self.net_worth <= 0
            reward = self._get_reward()
            self.current_step += 1
            obs = self._next_observation()
        
        return obs, reward, done, {}
    
    def _render_to_file(self, filename='render.txt'):

        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        file = open(filename, 'a+')

        #file.write(f'Step: {self.current_step}\n')
        if self.actual_action == 0: #20210522
            file.write(f' BUY \n')
        elif self.actual_action == 1:
            file.write(f' SELL \n')
        else:
            file.write(f' HOLD \n')

        file.close()
        
    def reset(self):
        # Reset the state of the environment to an initial state
        
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

    def render(self, mode='human', close=False, **kwargs):
        # Render the environment to the screen
        #profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        

        if self.current_step == 0:
            self.current_step = len(df)

        print(f'Step: {self.current_step}')
        print(self.df.loc[self.current_step-1, "date"])
        print("Close at:", self.df.loc[self.current_step-1, "close"])

        if self.actual_action == 0:
            print('BUY!! at:', self.df.loc[self.current_step-1, "close"])
            print('Amount: ', self.shares_held * self.df.loc[self.current_step-1, "close"], "(", self.shares_held, ")")
        if self.actual_action  == 1:
            print('SELD!! at:', self.df.loc[self.current_step-1, "close"])
            print('Amount: ', self.shares_held* self.df.loc[self.current_step-1, "close"], "(", self.shares_held, ")")
        if self.actual_action == 2:
            print('HOLD!!!')
        print()
        print(f'Fee: {self.fee}')
        print(f'Tax: {self.tax}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        #print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        #print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        #print(f'Profit: {profit}')
        #print('\n')


        #print('Buy cost: ',self.buy_cost)
        #print('Sell profit: ', self.cumulation_profit)
        #print('Profit: ', self.cumulation_profit - self.buy_cost )

        print("Final Assets: ", self.balance + self.shares_held * self.df.loc[self.current_step-1, "close"])
        print("")


        self._render_to_file(kwargs.get('filename', 'render.txt'))
        print('====================================================\n')
        


    
    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None