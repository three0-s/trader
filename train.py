from learner import dqn_learning, OptimizerSpec
from utils.schedule import LinearSchedule
from model import Dueling_DQN
from envs.env import CryptoMarketEnv
from utils.wrapper import get_wrapper_by_name, get_env
import torch.optim as optim
import torch
from torchinfo import summary

# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 10000000
FRAME_HISTORY_LEN = 64
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 1e-4
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(2000000, 0.1)
LEARNING_STARTS = 100000
DATA_DIR = "/mnt/won/data"
RENDER_DIR = "render"
STEPS = 10e8
EMB_DIM=256
N_STOCK=1
NUM_HEADS=4
WEIGHT_DECAY=1e-5
NUM_LAYERS=6

def train(env, num_timesteps):
    
    optimizer = OptimizerSpec(
        constructor=optim.AdamW,
        kwargs=dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    )
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    dqn_learning(
        env=env,
        optimizer_spec=optimizer,
        device=device,
        q_func=Dueling_DQN,
        emb_dim=EMB_DIM,
        n_stocks=N_STOCK,
        num_head=NUM_HEADS,
        num_layers=NUM_LAYERS,

        exploration=EXPLORATION_SCHEDULE,
        stopping_criterion=num_timesteps,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
    )
    env.close()


if __name__ == "__main__":
    env = CryptoMarketEnv(data_dir=DATA_DIR,
                          n_stock=14,
                          SL=0.25,
                          TP=0.7,
                          render_dir=RENDER_DIR)
    env = get_env(env, 928, RENDER_DIR)
    print("="*40)
    print("Train Start! ...".center(40))
    print("="*40)
    B, N, L, F, D = BATCH_SIZE, 1, FRAME_HISTORY_LEN, 16, EMB_DIM

    A = 9
    a = torch.zeros(B, N, L, F)
    model = Dueling_DQN(F, A, D, N, 8, NUM_LAYERS, L)
    summary(model, (B, N, L, F))
    train(env, STEPS)