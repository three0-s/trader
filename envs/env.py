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

ACTION_DICT = {
    NOOP: "No OP",
    LONG1X: "Long 1x",
    LONG2X: "Long 2x",
    SELL_L1X: "Sell Long 1x",
    SELL_L2X: "Sell Long 2x",
    SHORT1X: "Short 1x",
    SHORT2X: "Short 2x",
    SELL_S1X: "Sell Short 1x",
    SELL_S2X: "Sell Short 2x",
}


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
    
    def get_net_profit_rate(self):
        net_worth = self.get_networth()
        return (net_worth-INIT_BALANCE) / INIT_BALANCE

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
        self.cur=random.randint(0, len(self.current_map_df)//2)
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
        if (self.cur >= len(self.current_map_df)-2):
            return True
        networth = self.get_networth()
        if (networth <= INIT_BALANCE*(1-self.SL) or networth >= INIT_BALANCE*(1+self.TP)):
            return True
        
        return False
    

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)
        done = self._isend()
        # before_done = (self.cur >= len(self.current_map_df)-2)
        if done:
            # SELL ALL HOLDING SHARES
            # self.cur = len(self.current_map_df)-2 # time travel
            action = torch.zeros(self.action_space.shape)
            action[SELL_L1X]=1e8
            r = self._take_action(action)
            r=0 if r < 0 else r
            reward+=r

            action = torch.zeros(self.action_space.shape)
            action[SELL_L2X]=1e8
            r = self._take_action(action)
            r=0 if r < 0 else r
            reward+=r

            action = torch.zeros(self.action_space.shape)
            action[SELL_S1X]=1e8
            r = self._take_action(action)
            r=0 if r < 0 else r
            reward+=r

            action = torch.zeros(self.action_space.shape)
            action[SELL_S2X]=1e8
            r = self._take_action(action)
            r=0 if r < 0 else r
            reward+=r

        self.cur+=1
        obs = self._next_observation()
        return obs, reward, done, {}
    

    def _take_action(self, action):
        action_type = torch.argmax(action)

        if action_type == LONG1X or action_type == LONG2X:
            # Set the current price to the highest price within the time step
            current_price = self.current_map_df['High'].iloc[self.cur]
        elif action_type == SELL_L1X or action_type == SELL_L2X:
            # Set the current price to the lowest price within the time step
            current_price = self.current_map_df['Low'].iloc[self.cur]
        if action_type == SHORT1X or action_type == SHORT2X:
            # Set the current price to the lowest price within the time step
            current_price = self.current_map_df['Low'].iloc[self.cur]
        elif action_type == SELL_S1X or action_type == SELL_S2X:
            # Set the current price to the highest price within the time step
            current_price = self.current_map_df['High'].iloc[self.cur]
        elif action_type == NOOP:
            current_price = 1
        # assert all the elements be semi-positive
        action -= torch.min(action)
        action = torch.abs(action)
        # amount = action[action_type]
        tot_price = action[action_type]/(torch.sum(action)+ eps) * self.account.balance * 0.05 
        amount = torch.abs(tot_price/current_price)
        assert amount >= 0, f"Trading units must be semi-positive; Got Total Price: {tot_price}, Acc Balance: {self.account.balance}, \
                             Current Price: {current_price} and amount: {amount}"
        # reward = self.get_net_profit_rate()
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
                amount = self.CPS[0][1]
            if amount > 0:
                # balance update & reward
                self.account.deposit(current_price * amount * (1-self.fee))
                # reward += (current_price-self.CPS[0][0]) / self.CPS[0][0] * (1-self.fee)
                reward += self.get_net_profit_rate()
                # CPS state update
                if (amount < self.CPS[0][1]):
                    self.CPS[0] = (self.CPS[0][0], self.CPS[0][1]-amount)
                else:
                    self.CPS[0] = (0, 0)
            else:
                # reward -= 0.01/100
                pass

        elif action_type==SELL_L2X:
            if (amount > self.CPS[1][1]):
                amount = self.CPS[1][1]

            if amount > 0:
                # balance update & reward
                profit = (current_price-self.CPS[1][0])*(2)*amount * (1-self.fee)
                total_shares = self.CPS[1][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                reward += profit / (self.CPS[1][0] * amount)
                # reward += self.get_net_profit_rate()
                # CPS state update
                if (amount < self.CPS[1][1]):
                    self.CPS[1] = (self.CPS[1][0], self.CPS[1][1]-amount)
                else:
                    self.CPS[1] = (0, 0)
            else:
                # reward -= 0.01/100
                pass

        elif action_type==SHORT1X:
            if self.account.withdraw(tot_price):
                self.CPS[2] = ((self.CPS[2][0]*self.CPS[2][1] + tot_price)/(self.CPS[2][1] + amount + eps),
                               self.CPS[2][1] + amount)
                
        elif action_type==SHORT2X:
            if self.account.withdraw(tot_price):
                self.CPS[3] = ((self.CPS[3][0]*self.CPS[3][1] + tot_price)/(self.CPS[3][1] + amount+ eps),
                               self.CPS[3][1] + amount)
                
        elif action_type==SELL_S1X:
            if (amount > self.CPS[2][1]):
                amount = self.CPS[2][1]

            if amount > 0:
                # balance update & reward
                profit = (self.CPS[2][0]-current_price) * (1) * amount * (1-self.fee)
                total_shares = self.CPS[2][0] * amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                reward += profit / (self.CPS[2][0] * amount+ eps)
                # reward = self.get_net_profit_rate()
                
                # CPS state update
                if (amount < self.CPS[2][1]):
                    self.CPS[2] = (self.CPS[2][0], self.CPS[2][1]-amount)
                else:
                    self.CPS[2] = (0, 0)
            else:
                # reward -= 0.01/100
                pass

        elif action_type==SELL_S2X:
            if (amount > self.CPS[3][1]):
                amount = self.CPS[3][1]

            if amount > 0:
                # balance update & reward
                profit = (self.CPS[3][0]-current_price) * (2) * amount * (1-self.fee)
                total_shares = self.CPS[3][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                reward += profit / (self.CPS[3][0] * amount+ eps)
                # reward += self.get_net_profit_rate()
                # CPS state update
                if (amount < self.CPS[3][1]):
                    self.CPS[3] = (self.CPS[3][0], self.CPS[3][1]-amount)
                else:
                    self.CPS[3] = (0, 0)
            else:
                # reward -= 0.01/100
                pass
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
        profit_rate = self.get_net_profit_rate()*100
        color = 'g' if profit_rate >0 else 'r'
        ay.text(0, 0.7, f"Net Profit Rate: {profit_rate:.2f}%", color=color)
        mpl_finance.candlestick2_ohlc(ax, pre['Open'], pre['High'], pre['Low'], pre['Close'], width=0.5, colorup='r', colordown='b')

        plt.savefig(os.path.join(self.current_render_dir, f"{self.cur}.png"))
        plt.close()
        