from model import Dueling_DQN   
from envs.env import CryptoMarketEnv
import torch 
from collections import deque


BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 10000000
FRAME_HISTORY_LEN = 64
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 1e-4
ALPHA = 0.95
EPS = 0.01
LEARNING_STARTS = 100000
DATA_DIR = "/mnt/won/data"
RENDER_DIR = "/mnt/won/render/"
STEPS = 10e8
EMB_DIM=256
N_STOCK=1
NUM_HEADS=4
NUM_LAYERS=6

model_path = "/mnt/won/models/dueling_1700000_Wed_May_10_13:08:00_2023.model"



if __name__ == "__main__":
    env = CryptoMarketEnv(data_dir=DATA_DIR,
                        n_stock=14,
                        SL=0.02,
                        TP=0.04,
                        render_dir=RENDER_DIR)
    F = env.observation_space.shape[0]
    N = 1
    input_shape = (N, FRAME_HISTORY_LEN, F)
    in_channels = input_shape[2]
    num_actions = env.action_space.shape[0]

    device = torch.device("cuda:1")
    Q = Dueling_DQN(in_channels, num_actions, EMB_DIM, N_STOCK, NUM_HEADS, NUM_LAYERS, FRAME_HISTORY_LEN)
    Q.load_state_dict(torch.load(model_path))
    Q.eval().to(device)

    with torch.no_grad():
        obs = env.reset()
        obs_concat = deque(maxlen=FRAME_HISTORY_LEN)
        tot_reward = 0
        i = 0
        with torch.no_grad():
            while True:
                i+=1
                if type(obs)!=torch.Tensor:
                    obs = torch.from_numpy(obs)
                obs = obs.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
                obs_concat.append(obs)
                if i <= FRAME_HISTORY_LEN:
                    action = torch.zeros(num_actions,).cpu().squeeze()
                    action[0]=1.0 #do nothing
                else:
                    s = torch.cat(list(obs_concat), dim=2).to(device)
                    action = Q(s).cpu().squeeze() #(16, )
                
                obs, rewards, done, info = env.step(action)
                if (done):
                    tot_reward = env.get_net_profit_rate()
                    env.render()
                    break
                env.render()
        print(f"Total net profit rate: {tot_reward*100:.2f}%")
        