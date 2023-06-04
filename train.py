from learner import dqn_learning, OptimizerSpec
from utils.schedule import LinearSchedule
from model import Dueling_DQN
from envs.env import CryptoMarketEnv, ACTION_DICT
from utils.logger import Logger
from utils.wrapper import get_wrapper_by_name, get_env
import torch.optim as optim
import torch
from torchinfo import summary
import sys 



# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000000
FRAME_HISTORY_LEN = 64
TARGET_UPDATE_FREQ = 50000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 1e-4
EXPLORATION_SCHEDULE = LinearSchedule(5000000, 0.25)
LEARNING_STARTS = 100000
DATA_DIR = "/root/won/data"
RENDER_DIR = "render"
STEPS = 10e8
EMB_DIM=256
N_STOCK=1
NUM_HEADS=8
WEIGHT_DECAY=1e-4
NUM_LAYERS=10
SL = 0.01
TP = 0.03

CUDA_NO = 0

def train(env, num_timesteps, device):
    optimizer = OptimizerSpec(
        constructor=optim.AdamW,
        kwargs=dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    )
    # Set the logger
    logger = Logger('./logs')

    logger.LogAndPrint("="*40)
    logger.LogAndPrint("TRAINING CONFIGS".center(40))
    logger.LogAndPrint("")
    logger.LogAndPrint("Batch Size".ljust(20)+f"{BATCH_SIZE}".rjust(20))
    logger.LogAndPrint("Replay Buffer Size".ljust(20)+f"{REPLAY_BUFFER_SIZE}".rjust(20))
    logger.LogAndPrint("Time Sequence".ljust(20)+f"{FRAME_HISTORY_LEN}".rjust(20))
    logger.LogAndPrint("Learning Rate".ljust(20)+f"{LEARNING_RATE:.4f}".rjust(20))
    logger.LogAndPrint("Model Dimension".ljust(20)+f"{EMB_DIM}".rjust(20))
    logger.LogAndPrint("# of Attention Heads".ljust(20)+f"{NUM_HEADS}".rjust(20))
    logger.LogAndPrint("# of Attention Layer".ljust(20)+f"{NUM_LAYERS}".rjust(20))
    logger.LogAndPrint("Stop Loss".ljust(20)+f"{SL:.3f}".rjust(20))
    logger.LogAndPrint("Take Profit".ljust(20)+f"{TP:.3f}".rjust(20))
    logger.LogAndPrint("="*40)
    sys.stdout.flush()

    dqn_learning(
        env=env,
        logger=logger,
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
                          SL=SL,
                          TP=TP,
                          render_dir=RENDER_DIR)
    env = get_env(env, 415, RENDER_DIR)
    device = 'cpu' if not torch.cuda.is_available() else torch.device(f'cuda:{CUDA_NO}')
    
    print("="*40)
    print("Train Start! ...".center(40))
    print("="*40)
    A = len(ACTION_DICT.keys())
    B, N, L, F, D = BATCH_SIZE, 1, FRAME_HISTORY_LEN, 6+len(env.CPS)*2, EMB_DIM
   
    a = torch.zeros(B, N, L, F).to(device)
    model = Dueling_DQN(F, A, D, N, NUM_HEADS, NUM_LAYERS, L).to(device)
    summary(model, (B, N, L, F))
    train(env, STEPS, device=device)