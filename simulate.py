from model import Dueling_DQN   
from envs.env import CryptoMarketEnv
import torch 
from collections import deque



REPLAY_BUFFER_SIZE = 10000000
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 1e-4

DATA_DIR = "/root/won/data"
RENDER_DIR = "/root/won/render/"



FRAME_HISTORY_LEN = 128

LEARNING_STARTS = 50000

EMB_DIM=512
N_STOCK=1
NUM_HEADS=8
NUM_LAYERS=8

model_path = "/root/won/models/dueling_500000_Fri_May_12_14:43:07_2023.model"



if __name__ == "__main__":
    env = CryptoMarketEnv(data_dir=DATA_DIR,
                        n_stock=14,
                        SL=0.015,
                        TP=0.02,
                        render_dir=RENDER_DIR)
    F = env.observation_space.shape[0]
    N = 1
    input_shape = (N, FRAME_HISTORY_LEN, F)
    in_channels = input_shape[2]
    num_actions = env.action_space.shape[0]

    device = torch.device("cuda:0")
    Q = Dueling_DQN(in_channels, num_actions, EMB_DIM, N_STOCK, NUM_HEADS, NUM_LAYERS, FRAME_HISTORY_LEN)
    Q.load_state_dict(torch.load(model_path))
    Q.to(device)
#     Q.eval().to(device)

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
        