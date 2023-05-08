import os
import numpy as np
import pandas as pd
import random
import gym
from gym import spaces
from .account import VirtualAccount
from glob import glob
from matplotlib import pyplot as plt
import mpl_finance
from time import time
import torch


NOOP=0
LONG1X=1
SELL_L1X=2
LONG2X=3
SELL_L2X=4
SHORT2X=5
SELL_S2X=6
SHORT1X=7
SELL_S1X=8


eps = 1e-7
MAX_BALANCE = 10e9
INIT_BALANCE = 10e5
class CryptoMarketEnv(gym.Env): 
    metadata = {'render.modes': ['human']}
    def __init__(self, data_dir, n_stock, SL:float, TP:float, render_dir):
        super(CryptoMarketEnv, self).__init__()
        self.state = None
        self.fee = 0.5/100 # 0.5% fee
        self.data_dir = data_dir
        self.n_stock = n_stock
        
        # Stop Loss and Take Profit
        assert (SL > 0 and SL < 1 and TP > 0), "SL should be less than 1"
        self.SL = SL
        self.TP = TP
        self.render_dir = render_dir
        self.game_maps = dict()
        for i in range(n_stock):
           self.game_maps[i]=glob(os.path.join(data_dir, str(i), "*.csv"))

        self.current_map_df = None
        self.cur = 0
        # (CPS, # of shares) 
        self.CPS = [(0, 0), # Long 1x 
                    (0, 0), # Long 2x 
                    (0, 0), # Short 1x 
                    (0, 0)] # Short 2x  
                              
        self.action_space = spaces.Box(low=0, high=1000000, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000000, shape=(16,), dtype=np.float32) 
        # state vector 
        #         s_t = [Count Open High Low Close Vol VWAP Balance C_L1 S_L1 C_L2 S_L2 C_S1 S_S1 C_S2 S_S2]
        self.reward_range = (-MAX_BALANCE, MAX_BALANCE) 
        self.account = VirtualAccount(INIT_BALANCE)
        
    '''
    def seed(self, seed=None):
        pass

    def step(self, action):
        assert self.action_space.contains(action)
        self.state = self.next(self.df)  # (price_percentage, volume_percentage, price, finished)
        
        reward = 0
        observation = np.array([[self.state[0], self.state[1]]]).reshape((1, 2, 1))
        
        finished = self.state[3]
        info = {}
        info["price"] = self.state[2]

        if finished and (info["price"] != 0):
          if self.purchased:
            reward = ((info["price"] * self.commision / (self.purchased_price + eps)) - 1) * 1000
            self.purchased = False
          print(self.buy_num)

          return observation, reward, finished, info 

        if (action == ACTION_BUY) and (not self.purchased):
          if (self.account.withdraw(info["price"])):
            self.purchased = True
            self.buy_num += 1
            self.purchased_price = info["price"]

        elif (action == ACTION_SELL) and self.purchased:
          if self.account.deposit(info["price"]):
            self.purchased = False
            reward = ((info["price"] * self.commision/ (self.purchased_price + eps)) - 1) * 1000 #reward corresponds to the revenue ratio
            self.reward_list.append(reward)

        return observation, reward, finished, info 


    def get_data(self, train=True):
      if train:
        data = random.choice(self.companies)
        return pd.read_csv(data)
        
      else:
        pass


    def next(self, df):
      finished = False
      if self.index == (len(df) - 1):
        finished = True
      
      elif self.index > (len(df) - 1):
        finished = True
        return (0, 0, 0, finished)
      
      price_percentage = (1 - df.iloc[self.index]["체결가"] / df.iloc[0]["체결가"]) * 100
      volume_percentage = (1 - df.iloc[self.index]["거래량"] / df.iloc[0]["거래량"]) * 100
      price = df.iloc[self.index]["체결가"]

      self.index += 1

      return (price_percentage, volume_percentage, price, finished)
    '''


    def reset(self):
        self.CPS = [(0, 0), # Long 1x 
                    (0, 0), # Long 2x 
                    (0, 0), # Short 1x 
                    (0, 0)] # Short 2x    
        map_no = random.randint(0, 13)
        dfname = random.choice(self.game_maps[map_no])
        self.current_map_df = pd.read_csv(dfname)
        del self.current_map_df['timestamp']
        now = str(int(time()))
        self.current_render_dir = os.path.join(self.render_dir, str(map_no), os.path.basename(dfname).replace('.csv', ''), now)
        self.cur=random.randint(0, len(self.current_map_df)-60)
        self.account.set_balance(INIT_BALANCE)

        return self._next_observation()
    

    def _next_observation(self):
        obs = self.current_map_df.iloc[self.cur].to_numpy() # (7,)
        cps = []
        for t in self.CPS:
            t0, t1 = t
            if (type(t0)==torch.Tensor):
                t0 = t0.detach().cpu().numpy()
            if (type(t1)==torch.Tensor):
                t1 = t1.detach().cpu().numpy()
            cps.append(t0)
            cps.append(t1)
        if type(self.account.balance)==torch.Tensor:
            self.account.balance = self.account.balance.detach().cpu().numpy()
        acc_state = np.array([self.account.balance, *cps])  # (9,)
        out = np.concatenate([obs, acc_state])
        return out
    
    
    def get_networth(self):
        current_price = self.current_map_df['Low'].iloc[self.cur]
        profit=0
        # Long
        for i in range(2):
            if type(self.CPS[i][0])==torch.Tensor:
                cps = self.CPS[i][0].detach().cpu()
            else:
                cps = self.CPS[i][0]
            if type(self.CPS[i][1])==torch.Tensor:
                share = self.CPS[i][1].detach().cpu()
            else:
                share = self.CPS[i][1]
            profit += (current_price-cps)*(i+1)*share

        # Short
        for i in range(2):
            profit += (self.CPS[i+2][0]-current_price)*(i+1)*self.CPS[i+2][1]

        total_shares=0
        for i in range(4):
            total_shares += self.CPS[i][0]*self.CPS[i][1]
        
        return self.account.balance + profit + total_shares


    def _isend(self):
        if (self.cur >= len(self.current_map_df)):
            return True
        networth = self.get_networth()
        if (networth <= INIT_BALANCE*(1-self.SL) or networth >= INIT_BALANCE*(1+self.TP)):
            return True
        
        return False
    

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)
        self.cur+=1
        done = self._isend()
        if done:
            self.cur=0
        obs = self._next_observation()
        return obs, reward, done, {}
    

    def _take_action(self, action):
        # Set the current price to the highest price within the time step
        current_price = self.current_map_df['High'].iloc[self.cur]
        action_type = torch.argmax(action)
        # amount = action[action_type]
        amount = action[action_type]/torch.sum(action) * self.account.balance * 0.1
        tot_price = current_price*amount
        reward = 0

        if action_type==NOOP:
            pass
        elif action_type==LONG1X:
            if self.account.withdraw(tot_price):
                self.CPS[0] = ((self.CPS[0][0]*self.CPS[0][1] + tot_price)/(self.CPS[0][1] + amount),
                               self.CPS[0][1] + amount)

        elif action_type==LONG2X:
            if self.account.withdraw(tot_price):
                self.CPS[1] = ((self.CPS[1][0]*self.CPS[1][1] + tot_price)/(self.CPS[1][1] + amount),
                               self.CPS[1][1] + amount)
                
        elif action_type==SELL_L1X:
            if (amount > self.CPS[0][1]):
                reward -= 0.1
                amount = self.CPS[0][1]
            if amount > 0:
                # balance update & reward
                self.account.deposit(current_price * amount * (1-self.fee))
                reward += (current_price-self.CPS[0][0]) / self.CPS[0][0] * (1-self.fee)
                # CPS state update
                if (amount < self.CPS[0][1]):
                    self.CPS[0] = (self.CPS[0][0], self.CPS[0][1]-amount)
                else:
                    self.CPS[0] = (0, 0)

        elif action_type==SELL_L2X:
            if (amount > self.CPS[1][1]):
                reward -= 0.1
                amount = self.CPS[1][1]
            if amount > 0:
                # balance update & reward
                profit = (current_price-self.CPS[1][0])*(2)*amount * (1-self.fee)
                total_shares = self.CPS[1][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                reward += profit / (self.CPS[1][0] * amount)

                # CPS state update
                if (amount < self.CPS[1][1]):
                    self.CPS[1] = (self.CPS[1][0], self.CPS[1][1]-amount)
                else:
                    self.CPS[1] = (0, 0)

        elif action_type==SHORT1X:
            if self.account.withdraw(tot_price):
                self.CPS[2] = ((self.CPS[2][0]*self.CPS[2][1] + tot_price)/(self.CPS[2][1] + amount),
                               self.CPS[2][1] + amount)
                
        elif action_type==SHORT2X:
            if self.account.withdraw(tot_price):
                self.CPS[3] = ((self.CPS[3][0]*self.CPS[3][1] + tot_price)/(self.CPS[3][1] + amount),
                               self.CPS[3][1] + amount)
                
        elif action_type==SELL_S1X:
            if (amount > self.CPS[2][1]):
                reward -= 0.1
                amount = self.CPS[2][1]
            if amount > 0:
                # balance update & reward
                profit = (self.CPS[2][0]-current_price) * (1) * amount * (1-self.fee)
                total_shares = self.CPS[2][0] * amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                reward += profit / (self.CPS[2][0] * amount)

                
                # CPS state update
                if (amount < self.CPS[2][1]):
                    self.CPS[2] = (self.CPS[2][0], self.CPS[2][1]-amount)
                else:
                    self.CPS[2] = (0, 0)

        elif action_type==SELL_S2X:
            if (amount > self.CPS[3][1]):
                reward -= 0.1
                amount = self.CPS[3][1]

            if amount > 0:
                # balance update & reward
                profit = (self.CPS[3][0]-current_price) * (2) * amount * (1-self.fee)
                total_shares = self.CPS[3][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                reward += profit / (self.CPS[3][0] * amount)

                # CPS state update
                if (amount < self.CPS[3][1]):
                    self.CPS[3] = (self.CPS[3][0], self.CPS[3][1]-amount)
                else:
                    self.CPS[3] = (0, 0)

        return reward



    def render(self, mode='human', close=False):
        position = ['Long 1x', 'Long 2x', 'Short 1x', 'Short 2x']
        try:
            os.makedirs(self.current_render_dir)
        except OSError:
            pass

        st = self.cur-15 if self.cur >= 15  else 0
        pre = self.current_map_df.iloc[st:self.cur]
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_ylabel("USD")
        plt.grid()
        ay = fig.add_subplot(1, 2, 2)
        ay.axis('off')

        prev = -1
        for i in range(4):
            if self.CPS[i][1] != 0:
                ay.text(0, 0.1+(prev+1)/20, f"{position[i]}] CPS: {self.CPS[i][0]:.2f}$      {self.CPS[i][1]:.2f}")
                prev += 1
        ay.text(0, 0.7, f"Net Profit Rate: {(self.get_networth()-INIT_BALANCE)*100/INIT_BALANCE:.2f}%", color='g')
        mpl_finance.candlestick2_ohlc(ax, pre['Open'], pre['High'], pre['Low'], pre['Close'], width=0.5, colorup='r', colordown='b')

        plt.savefig(os.path.join(self.current_render_dir, f"{self.cur}.png"))
        plt.close()
        