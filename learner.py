"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch
import sys
import os
import itertools
import numpy as np
import random
from torch import nn
from collections import namedtuple
from utils.replay_buffer import ReplayBuffer
from model import Dueling_DQN
from utils.schedule import LinearSchedule
import time
from utils.logger import Logger
from envs.env import CryptoMarketEnv
from gym import spaces
from utils.wrapper import get_wrapper_by_name

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


def to_np(x:torch.Tensor):
    return x.detach().cpu().numpy() 

def dqn_learning(env:CryptoMarketEnv,
          logger:Logger,
          optimizer_spec,
          device,
          q_func=Dueling_DQN,
          emb_dim=256,
          n_stocks=1,
          num_head=8,
          num_layers=3,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=64,
          target_update_freq=10000,
          ):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    env_id: string
        gym environment id for model saving.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == spaces.Box
    assert type(env.action_space)      == spaces.Box

    ###############
    # BUILD MODEL #
    ###############
    F = env.observation_space.shape[0]
    N = 1
    input_shape = (N, frame_history_len, F)
    in_channels = input_shape[2]
    num_actions = env.action_space.shape[0]
    
    # define Q target and Q 
    Q = q_func(in_channels, num_actions, emb_dim, n_stocks, num_head, num_layers, frame_history_len).to(device)
    Q_target = q_func(in_channels, num_actions, emb_dim, n_stocks, num_head, num_layers, frame_history_len).to(device)

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    
    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, num_actions)

    
    #  Huber Loss 
    objective = nn.SmoothL1Loss(reduction='none')
    
    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000
    SAVE_MODEL_EVERY_N_STEPS = 100000


    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion < t:
            break

        ### 2. Step the env and store the transition
        # store last frame, returned idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = torch.rand(env.action_space.shape).to(torch.float32).detach().cpu()
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = torch.from_numpy(observations).unsqueeze(0).to(device, torch.float32)
                with torch.no_grad():
                    action = Q(obs).cpu().squeeze() #(9, )
                # action = ((q_value_all_actions).max(1)[1])[0]
            else:
                action = torch.rand(env.action_space.shape).to(torch.float32).detach().cpu()

        obs, reward, done, info = env.step(action)
        if type(reward) != torch.Tensor:
            reward = torch.Tensor([reward])
        # clipping the reward, noted in nature paper
        reward = torch.clip(reward, -10, 10).detach().cpu()

        # store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action.detach().cpu().numpy(), reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()

        # update last_obs
        last_obs = obs
        ### 3. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = torch.from_numpy(obs_t).to(device, torch.float32)
            act_t = torch.from_numpy(act_t).to(device, torch.float32)
            rew_t = torch.from_numpy(rew_t).to(device, torch.float32)
            obs_tp1 = torch.from_numpy(obs_tp1).to(device, torch.float32)
            done_mask = torch.from_numpy(done_mask).to(device, torch.float32)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
           
            q_values = Q(obs_t).squeeze() # (B, num_action)
            q_s_a = q_values.gather(1, torch.argmax(act_t, dim=1).unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            
            # ---------------
            #   regular DQN
            # ---------------

            # get the Q values for best actions in obs_tp1 
            # based off frozen Q network
            # max(Q(s', a', theta_i_frozen)) wrt a'
            with torch.no_grad():
                q_tp1_values = Q_target(obs_tp1).detach()
                q_s_a_prime, a_prime  = q_tp1_values.max(dim=2)
                q_s_a_prime = q_s_a_prime.squeeze()
                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime 

            # Compute Bellman error
            # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
#             error = rew_t + gamma * q_s_a_prime - q_s_a
            loss = objective(rew_t + gamma * q_s_a_prime, q_s_a)
            # backwards pass
            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(Q.parameters(), 100)
            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if t % LOG_EVERY_N_STEPS == 0:
                mloss = np.mean(to_np(loss))
                logger.scalar_summary("Training Loss (Huber)", mloss, t+1)
                logger.LogAndPrint(f"Training Loss (Huber) {mloss:.3f}")
            
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # # (2) Log values and gradients of the parameters (histogram)
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in Q.named_parameters():
                    tag = tag.replace('.', '/')
                    try:
                        logger.histo_summary(tag, to_np(value), t+1)
                        logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
                    except:
                        continue
            #####

        ### 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = 'dueling'
            model_save_path = "models/%s_%d_%s.model" %(add_str, t, str(time.ctime()).replace(' ', '_'))
            torch.save(Q.state_dict(), model_save_path)
 
        
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            logger.LogAndPrint("---------------------------------")
            logger.LogAndPrint("Timestep %d" % (t,))
            logger.LogAndPrint("learning started? %d" % (t > learning_starts))
            logger.LogAndPrint("mean reward (100 episodes) %f" % mean_episode_reward)
            logger.LogAndPrint("best mean reward %f" % best_mean_episode_reward)
            logger.LogAndPrint("episodes %d" % len(episode_rewards))
            logger.LogAndPrint("exploration %f" % exploration.value(t))
            logger.LogAndPrint("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': episode_rewards[-1],
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    'best_mean_episode_reward': best_mean_episode_reward
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)