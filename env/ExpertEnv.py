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

    def __init__(self, df, window_size, frame_bound):
        super(StockTradingEnv, self).__init__()
        
        # 基礎股價資訊
        self.df = df
        self.window_size = window_size
        self.last_step = None
        
        
        # 獎勵值範圍
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)

        # 只決定是否持有股票，要馬全買(資產竟可能轉換成股票)，要馬全賣(資產全換成現金)
        # 0: all cash
        # 1: all stock
        self.action_space = spaces.Discrete(2)

        # 觀測值Size

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
          
        '''
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(220,), dtype=np.float16)
            #low=0, high=1, shape=(180,), dtype=np.float16)'''
    def _process_data(self):
        raise NotImplementedError
    def _next_observation(self):
        #print(self.current_step)
        obs = self.df.iloc[self.current_step].values.tolist()
        return obs

    def _take_action(self, action):

        # 一般化的化建議取最高最低價之間隨意的價格，但為了比較其他組實驗上的方便，購入價格定為收盤價
        
        #print(self.current_step, len(self.df), self.df.loc[self.current_step, "date"])
        current_price = self.df.iloc[self.current_step]['close']
        #print(self.shares_held)
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
                
                self.last_step = self.current_step
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
            previous_price = self.df.iloc[self.current_step]["open"]
            
        #previous_price = self.df.iloc[self.last_step, "close"]
        current_price = self.df.iloc[self.current_step]["close"]
        
        RR = (current_price - previous_price) / previous_price
        
        
        if self.actual_action == 0:
            # buy
            reward = RR# - 0.001425
            
        elif self.actual_action == 1:
            # sell
            # 賣的時候就是看收益率/投資報酬率
            RoR = (self.sell_at_price - self.buy_at_price) / self.buy_at_price
            reward = RoR# - 0.001425 - 0.003
            
        elif self.actual_action == 2:
            # hold
            # RR正的時候盡量持有，負的時候就不要持有
            reward = RR
            
        return reward
    
    
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        #print(action)
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
        
        self.last_step = None
        
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
        '''
    def render(self, mode: str='human', **kwargs: Any) -> Any:
        if mode == 'simple_figure':
            return self._render_simple_figure(**kwargs)
        if mode == 'advanced_figure':
            return self._render_advanced_figure(**kwargs)
        return self.simulator.get_state(**kwargs)


    def _render_simple_figure(
        self, figsize: Tuple[float, float]=(14, 6), return_figure: bool=False
    ) -> Any:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            ax.plot(self.time_points, close_price, c=symbol_color, marker='.', label=symbol)

            buy_ticks = []
            buy_error_ticks = []
            sell_ticks = []
            sell_error_ticks = []
            close_ticks = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol, {})
                if order and not order['hold']:
                    if order['order_type'] == OrderType.Buy:
                        if order['error']:
                            buy_error_ticks.append(tick)
                        else:
                            buy_ticks.append(tick)
                    else:
                        if order['error']:
                            sell_error_ticks.append(tick)
                        else:
                            sell_ticks.append(tick)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    close_ticks.append(tick)

            tp = np.array(self.time_points)
            ax.plot(tp[buy_ticks], close_price[buy_ticks], '^', color='green')
            ax.plot(tp[buy_error_ticks], close_price[buy_error_ticks], '^', color='gray')
            ax.plot(tp[sell_ticks], close_price[sell_ticks], 'v', color='red')
            ax.plot(tp[sell_error_ticks], close_price[sell_error_ticks], 'v', color='gray')
            ax.plot(tp[close_ticks], close_price[close_ticks], '|', color='black')

            ax.tick_params(axis='y', labelcolor=symbol_color)
            ax.yaxis.tick_left()
            if j < len(self.trading_symbols) - 1:
                ax = ax.twinx()

        fig.suptitle(
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.legend(loc='right')

        if return_figure:
            return fig

        plt.show()


    def _render_advanced_figure(
            self, figsize: Tuple[float, float]=(1400, 600), time_format: str="%Y-%m-%d %H:%m",
            return_figure: bool=False
        ) -> Any:

        fig = go.Figure()

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))
        get_color_string = lambda color: "rgba(%s, %s, %s, %s)" % tuple(color)

        extra_info = [
            f"balance: {h['balance']:.6f} {self.simulator.unit}<br>"
            f"equity: {h['equity']:.6f}<br>"
            f"margin: {h['margin']:.6f}<br>"
            f"free margin: {h['free_margin']:.6f}<br>"
            f"margin level: {h['margin_level']:.6f}"
            for h in self.history
        ]
        extra_info = [extra_info[0]] * (self.window_size - 1) + extra_info

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            fig.add_trace(
                go.Scatter(
                    x=self.time_points,
                    y=close_price,
                    mode='lines+markers',
                    line_color=get_color_string(symbol_color),
                    opacity=1.0,
                    hovertext=extra_info,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.update_layout(**{
                f'yaxis{j+1}': dict(
                    tickfont=dict(color=get_color_string(symbol_color * [1, 1, 1, 0.8])),
                    overlaying='y' if j > 0 else None,
                    # position=0.035*j
                ),
            })

            trade_ticks = []
            trade_markers = []
            trade_colors = []
            trade_sizes = []
            trade_extra_info = []
            trade_max_volume = max([
                h.get('orders', {}).get(symbol, {}).get('modified_volume') or 0
                for h in self.history
            ])
            close_ticks = []
            close_extra_info = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol)
                if order and not order['hold']:
                    marker = None
                    color = None
                    size = 8 + 22 * (order['modified_volume'] / trade_max_volume)
                    info = (
                        f"order id: {order['order_id'] or ''}<br>"
                        f"hold probability: {order['hold_probability']:.4f}<br>"
                        f"hold: {order['hold']}<br>"
                        f"volume: {order['volume']:.6f}<br>"
                        f"modified volume: {order['modified_volume']:.4f}<br>"
                        f"fee: {order['fee']:.6f}<br>"
                        f"margin: {order['margin']:.6f}<br>"
                        f"error: {order['error']}"
                    )

                    if order['order_type'] == OrderType.Buy:
                        marker = 'triangle-up'
                        color = 'gray' if order['error'] else 'green'
                    else:
                        marker = 'triangle-down'
                        color = 'gray' if order['error'] else 'red'

                    trade_ticks.append(tick)
                    trade_markers.append(marker)
                    trade_colors.append(color)
                    trade_sizes.append(size)
                    trade_extra_info.append(info)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    info = []
                    for order in closed_orders:
                        info_i = (
                            f"order id: {order['order_id']}<br>"
                            f"order type: {order['order_type'].name}<br>"
                            f"close probability: {order['close_probability']:.4f}<br>"
                            f"margin: {order['margin']:.6f}<br>"
                            f"profit: {order['profit']:.6f}"
                        )
                        info.append(info_i)
                    info = '<br>---------------------------------<br>'.join(info)

                    close_ticks.append(tick)
                    close_extra_info.append(info)

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[trade_ticks],
                    y=close_price[trade_ticks],
                    mode='markers',
                    hovertext=trade_extra_info,
                    marker_symbol=trade_markers,
                    marker_color=trade_colors,
                    marker_size=trade_sizes,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[close_ticks],
                    y=close_price[close_ticks],
                    mode='markers',
                    hovertext=close_extra_info,
                    marker_symbol='line-ns',
                    marker_color='black',
                    marker_size=7,
                    marker_line_width=1.5,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

        title = (
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.update_layout(
            title=title,
            xaxis_tickformat=time_format,
            width=figsize[0],
            height=figsize[1],
        )

        if return_figure:
            return fig

        fig.show()


    def close(self) -> None:
        plt.close()'''
