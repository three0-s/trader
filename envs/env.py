import os
import numpy as np
import pandas as pd
import random
import gym
from gym import spaces
from .account import VirtualAccount
from glob import glob
from matplotlib import pyplot as plt
import mplfinance as mpf
from time import time
import torch


NOOP=0
LONG10X=1
SELL_L10X=2
LONG25X=3
SELL_L25X=4
LONG100X=5
SELL_L100X=6

SHORT100X=7
SELL_S100X=8
SHORT25X=9
SELL_S25X=10
SHORT10X=11
SELL_S10X=12



ID_COIN_NAME = ("Binance Coin",
                "Bitcoin", 
                "Bitcoin Cash",
                "Cardano",
                "Dogecoin",
                "EOS.IO",
                "Ethereum",
                "Ethereum Classic",
                "IOTA",
                "Litecoin",
                "Maker",
                "Monero",
                "Stellar",
                "TRON")
ACTION_DICT = {
    NOOP: "No OP",
    LONG10X: "Long 10x",
    LONG25X: "Long 25x",
    LONG100X: "Long 100x",

    SELL_L10X: "Sell Long 10x",
    SELL_L25X: "Sell Long 25x",
    SELL_L100X: "Sell Long 100x",

    SHORT10X: "Short 10x",
    SHORT25X: "Short 25x",
    SHORT100X: "Short 100x",

    SELL_S10X: "Sell Short 10x",
    SELL_S25X: "Sell Short 25x",
    SELL_S100X: "Sell Short 100x",
}


eps = 1e-7
MAX_BALANCE = 10e9
INIT_BALANCE = 10e5
GAME_LEN = 24*60
class CryptoMarketEnv(gym.Env): 
    metadata = {'render.modes': ['human']}
    def __init__(self, data_dir, n_stock, SL:float, TP:float, render_dir, test=False):
        super(CryptoMarketEnv, self).__init__()
        self.state = None
        self.fee = 0.5/100 # 0.5% fee
        self.data_dir = data_dir
        self.n_stock = n_stock
        self.test=test
        # Stop Loss and Take Profit
        assert (SL > 0 and SL < 1 and TP > 0), "SL should be less than 1"
        self.SL = SL
        self.TP = TP
        self.render_dir = render_dir
        self.game_maps = dict()
        for i in range(n_stock):
           self.game_maps[i]=glob(os.path.join(data_dir, str(i), "*.csv"))

        self.current_map_df = None
        self.map_no = None
        self.cur = 0
        self.init_cur=self.cur
        # (CPS, # of shares) 
        self.CPS = [(0, 0), # Long 10x 
                    (0, 0), # Long 25x 
                    (0, 0), # Long 100x

                    (0, 0), # Short 10x 
                    (0, 0), # Short 25x 
                    (0, 0)] # Short 100x    
                              
        self.action_space = spaces.Box(low=0, high=10000000, shape=(len(ACTION_DICT.keys()),), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10000000, shape=(5+1+len(self.CPS)*2,), dtype=np.float32) 
        # state vector 
        #         s_t = [Count Open High Low Close Vol VWAP Balance C_L1 S_L1 C_L2 S_L2 C_S1 S_S1 C_S2 S_S2]
        self.reward_range = (-MAX_BALANCE, MAX_BALANCE) 
        self.account = VirtualAccount(INIT_BALANCE)
    
    def get_net_profit_rate(self):
        net_worth = self.get_networth()
        return (net_worth-INIT_BALANCE) / INIT_BALANCE

    def reset(self, no=None):
        self.CPS = [(0, 0), # Long 10x 
                    (0, 0), # Long 25x 
                    (0, 0), # Long 100x

                    (0, 0), # Short 10x 
                    (0, 0), # Short 25x 
                    (0, 0)] # Short 100x    
        self.map_no = random.randint(0, 13)
        if no != None:
            self.map_no = no
        dfname = random.choice(self.game_maps[self.map_no])
        self.current_map_df = pd.read_csv(dfname)
        
        self.current_map_df['timestamp'] = pd.to_datetime(self.current_map_df['timestamp'], unit='s')
        self.current_map_df = self.current_map_df.set_index(['timestamp'])
        
        now = str(int(time()))
        self.current_render_dir = os.path.join(self.render_dir, str(self.map_no), os.path.basename(dfname).replace('.csv', ''), now)
        endpoint = len(self.current_map_df)-GAME_LEN-1 if (len(self.current_map_df)-GAME_LEN-1)>0 else 0
        self.cur=random.randint(0, endpoint)
        if self.test:
            self.cur=0
        self.init_cur=self.cur
        self.account.set_balance(INIT_BALANCE)

        return self._next_observation()
    

    def _next_observation(self):
        obs = self.current_map_df.iloc[self.cur].to_numpy() # (5,)
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
        acc_state = np.array([self.account.balance, *cps])  # (13,)
        out = np.concatenate([obs, acc_state])
        return out
    
    
    def get_networth(self):
        current_price = self.current_map_df['Close'].iloc[self.cur]
        profit=0
       
        # Long
        profit += (current_price-self.CPS[0][0])*(10)*self.CPS[0][1]
        profit += (current_price-self.CPS[1][0])*(25)*self.CPS[1][1]
        profit += (current_price-self.CPS[2][0])*(100)*self.CPS[2][1]

        # Short
        profit += (self.CPS[3][0]-current_price)*(10)*self.CPS[3][1]
        profit += (self.CPS[4][0]-current_price)*(25)*self.CPS[4][1]
        profit += (self.CPS[5][0]-current_price)*(100)*self.CPS[5][1]

        total_shares=0
        for i in range(6):
            total_shares += self.CPS[i][0]*self.CPS[i][1]
        
        return self.account.balance + profit + total_shares


    def _isend(self):
        if (self.cur >= len(self.current_map_df)-2 or self.cur-self.init_cur > GAME_LEN):
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
            action[SELL_L10X]=1
            r = self._take_action(action)
            
            action = torch.zeros(self.action_space.shape)
            action[SELL_L25X]=1
            r = self._take_action(action)

            action = torch.zeros(self.action_space.shape)
            action[SELL_L100X]=1
            r = self._take_action(action)

            action = torch.zeros(self.action_space.shape)
            action[SELL_S10X]=1
            r = self._take_action(action)

            action = torch.zeros(self.action_space.shape)
            action[SELL_S25X]=1
            r = self._take_action(action)

            action = torch.zeros(self.action_space.shape)
            action[SELL_S100X]=1
            r = self._take_action(action)
            reward += r
            
        self.cur+=1
        obs = self._next_observation()
        return obs, reward, done, {}
    

    def _take_action(self, action):
        action_type = torch.argmax(action)
        current_price = self.current_map_df['Close'].iloc[self.cur]
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

        elif action_type==LONG10X:
            if self.account.withdraw(tot_price):
                self.CPS[0] = ((self.CPS[0][0]*self.CPS[0][1] + tot_price)/(self.CPS[0][1] + amount),
                               self.CPS[0][1] + amount)

        elif action_type==LONG25X:
            if self.account.withdraw(tot_price):
                self.CPS[1] = ((self.CPS[1][0]*self.CPS[1][1] + tot_price)/(self.CPS[1][1] + amount),
                               self.CPS[1][1] + amount)
                
        elif action_type==LONG100X:
            if self.account.withdraw(tot_price):
                self.CPS[2] = ((self.CPS[2][0]*self.CPS[2][1] + tot_price)/(self.CPS[2][1] + amount),
                               self.CPS[2][1] + amount)
                
        elif action_type==SELL_L10X:
            # if (amount > self.CPS[0][1]):
            amount = self.CPS[0][1]
            if amount > 0:
                # balance update & reward
                profit = (current_price-self.CPS[0][0])*(10)*amount * (1-self.fee)
                total_shares = self.CPS[0][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                # CPS state update
                if (amount < self.CPS[0][1]):
                    self.CPS[0] = (self.CPS[0][0], self.CPS[0][1]-amount)
                else:
                    self.CPS[0] = (0, 0)
                reward += self.get_net_profit_rate()
            else:
                reward -= 0.005/100

        elif action_type==SELL_L25X:
            # if (amount > self.CPS[1][1]):
            amount = self.CPS[1][1]
            if amount > 0:
                # balance update & reward
                profit = (current_price-self.CPS[1][0])*(25)*amount * (1-self.fee)
                total_shares = self.CPS[1][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                # CPS state update
                if (amount < self.CPS[1][1]):
                    self.CPS[1] = (self.CPS[1][0], self.CPS[1][1]-amount)
                else:
                    self.CPS[1] = (0, 0)
                reward += self.get_net_profit_rate()
            else:
                reward -= 0.005/100

        elif action_type==SELL_L100X:
            amount = self.CPS[2][1]
            if amount > 0:
                # balance update & reward
                profit = (current_price-self.CPS[2][0])*(100)*amount * (1-self.fee)
                total_shares = self.CPS[2][0]*amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
                # CPS state update
                if (amount < self.CPS[2][1]):
                    self.CPS[2] = (self.CPS[2][0], self.CPS[2][1]-amount)
                else:
                    self.CPS[2] = (0, 0)
                reward += self.get_net_profit_rate()
            else:
                reward -= 0.005/100

        elif action_type==SHORT10X:
            if self.account.withdraw(tot_price):
                self.CPS[3] = ((self.CPS[3][0]*self.CPS[3][1] + tot_price)/(self.CPS[3][1] + amount + eps),
                               self.CPS[3][1] + amount)
                
        elif action_type==SHORT25X:
            if self.account.withdraw(tot_price):
                self.CPS[4] = ((self.CPS[4][0]*self.CPS[4][1] + tot_price)/(self.CPS[4][1] + amount+ eps),
                               self.CPS[4][1] + amount)
        
        elif action_type==SHORT100X:
            if self.account.withdraw(tot_price):
                self.CPS[5] = ((self.CPS[5][0]*self.CPS[5][1] + tot_price)/(self.CPS[5][1] + amount+ eps),
                               self.CPS[5][1] + amount)
                
        elif action_type==SELL_S10X:
            amount = self.CPS[3][1]

            if amount > 0:
                # balance update & reward
                profit = (self.CPS[3][0]-current_price) * (10) * amount * (1-self.fee)
                total_shares = self.CPS[3][0] * amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
               
                # CPS state update
                if (amount < self.CPS[3][1]):
                    self.CPS[3] = (self.CPS[3][0], self.CPS[3][1]-amount)
                else:
                    self.CPS[3] = (0, 0)
                reward += self.get_net_profit_rate()
            else:
                reward -= 0.005/100

        elif action_type==SELL_S25X:
            amount = self.CPS[4][1]

            if amount > 0:
                # balance update & reward
                profit = (self.CPS[4][0]-current_price) * (25) * amount * (1-self.fee)
                total_shares = self.CPS[4][0] * amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
 
                # CPS state update
                if (amount < self.CPS[4][1]):
                    self.CPS[4] = (self.CPS[4][0], self.CPS[4][1]-amount)
                else:
                    self.CPS[4] = (0, 0)
                reward += self.get_net_profit_rate()
            else:
                reward -= 0.005/100

        elif action_type==SELL_S100X:
            amount = self.CPS[5][1]

            if amount > 0:
                # balance update & reward
                profit = (self.CPS[5][0]-current_price) * (100) * amount * (1-self.fee)
                total_shares = self.CPS[5][0] * amount * (1-self.fee)
                self.account.deposit(profit+total_shares)
 
                # CPS state update
                if (amount < self.CPS[5][1]):
                    self.CPS[5] = (self.CPS[5][0], self.CPS[5][1]-amount)
                else:
                    self.CPS[5] = (0, 0)
                reward += self.get_net_profit_rate() 
            else:
                reward -= 0.005/100
                
        return reward



    def render(self, mode='human', close=False):
        position = ['Long 10x', 'Long 25x', 'Long 100x', 'Short 10x', 'Short 25x' , 'Short 100x']
        try:
            os.makedirs(self.current_render_dir)
        except OSError:
            pass

        st = self.cur-60 if self.cur >= 60  else 0
        pre = self.current_map_df.iloc[st:self.cur]
        fig = mpf.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1, style='binance')
        
        ax.spines['bottom'].set_visible(False)
        # ax.set_ylabel("Price")
        ax.grid(linestyle='--')
        ay = fig.add_subplot(1, 2, 2)
        ay.axis('off')
        
        prev = -1
        for i in range(6):
            if self.CPS[i][1] != 0:
                ay.text(0, 0.1+(prev+1)/20, f"{position[i]}] CPS: {self.CPS[i][0]:.2f}$      {self.CPS[i][1]:.2f}")
                prev += 1
        profit_rate = self.get_net_profit_rate()*100
        color = 'g' if profit_rate >0 else 'r'
        ay.text(0, 0.8, f"Net Profit Rate: {profit_rate:.2f}%", color=color)
        mpf.plot(pre, ax=ax, type='candle', axtitle=f'{ID_COIN_NAME[self.map_no]}/USD')
        
        fig.savefig(os.path.join(self.current_render_dir, f"{self.cur:06d}.png"))
        plt.close()
        