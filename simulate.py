from model import Dueling_DQN   
from envs.env import CryptoMarketEnv
import torch 
from collections import deque



REPLAY_BUFFER_SIZE = 10000000
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 1e-4

DATA_DIR = "/mnt/won/data"
RENDER_DIR = "/mnt/won/render/"



FRAME_HISTORY_LEN = 64

LEARNING_STARTS = 50000

EMB_DIM=256
N_STOCK=1
NUM_HEADS=8
NUM_LAYERS=6

model_path = "/mnt/won/models/dueling_2900000_Sat_May_13_13:05:27_2023.model"



if __name__ == "__main__":
    env = CryptoMarketEnv(data_dir=DATA_DIR,
                        n_stock=14,
                        SL=0.03,
                        TP=0.06,
                        render_dir=RENDER_DIR)
    F = env.observation_space.shape[0]
    N = 1
    input_shape = (N, FRAME_HISTORY_LEN, F)
    in_channels = input_shape[2]
    num_actions = env.action_space.shape[0]

    device = torch.device("cuda:0")
    Q = Dueling_DQN(in_channels, num_actions, EMB_DIM, N_STOCK, NUM_HEADS, NUM_LAYERS, FRAME_HISTORY_LEN)
    Q.load_state_dict(torch.load(model_path))
    # Q.to(device)
    Q.eval().to(device)
    mean_rewards =[]
    with torch.no_grad():
        obs = env.reset()
        obs_concat = deque(maxlen=FRAME_HISTORY_LEN)
        tot_reward = 0
        i = 0
       
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
            mean_rewards.append(rewards)
            if (done):
                tot_reward = env.get_net_profit_rate()
                # env.render()
                break
            # env.render()
    print(f"Total net profit rate: {tot_reward*100:.2f}%")
    print(f"max rewards: {max(mean_rewards)*100:.2f}%")
    