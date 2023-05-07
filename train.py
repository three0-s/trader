from learner import dqn_learning, OptimizerSpec
from utils.schedule import LinearSchedule
from model import Dueling_DQN
from envs.env import CryptoMarketEnv
from utils.wrapper import get_wrapper_by_name, get_env
import torch.optim as optim
import torch


# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 64
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
LEARNING_STARTS = 50000
DATA_DIR = "/Users/yewon/Documents/traderWon/envs/data"
RENDER_DIR = "/Users/yewon/Documents/traderWon/envs/test_render"
STEPS = 10e8


def train(env, num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    dqn_learning(
        env=env,
        optimizer_spec=optimizer,
        device=device,
        q_func=Dueling_DQN,
        emb_dim=256,
        n_stocks=1,
        num_head=8,
        num_layers=3,

        exploration=EXPLORATION_SCHEDULE,
        stopping_criterion=stopping_criterion,
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
    train(env, STEPS)