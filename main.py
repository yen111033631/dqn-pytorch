import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import pandas as pd

import gymnasium as gym

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    
def optimize_model(): # FIXME device
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    # actions = tuple((map(lambda a: torch.tensor([[a]]), batch.action))) # FIXME try to ignore
    # rewards = tuple((map(lambda r: torch.tensor([r]), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    state = torch.tensor(np.array(obs), dtype=torch.float32, device=device).view(1, 4, 84, -1) 
    return state

def train(env, n_episodes, render=False):
    
    rl_result = []
    
    for episode in range(n_episodes):
        # env reset
        obs, info = env.reset(seed=SEED+episode)
        state = get_state(obs)
        
        # init  
        total_reward = 0.0
        env_times_all = 0
        model_times_all = 0
        
        episode_start = time.time()
        
        
        for t in count():
            env_start = time.time()
            action = select_action(state)

            if render:
                env.render()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated  

            total_reward += reward                  

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action, next_state, reward)
            state = next_state
            
            env_end = time.time()

            if steps_done > INITIAL_MEMORY:
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            model_end = time.time()
            
            env_times_all += env_end - env_start
            model_times_all += model_end - env_end
            
            if done:
                episode_end = time.time()                
                one_episode_time = episode_end - episode_start
                
                rl_result.append([episode, t, total_reward, one_episode_time, env_times_all, model_times_all, steps_done])
                break
        if episode % 20 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
                
                df = pd.DataFrame(rl_result)
                df.columns = ["episode", "t", "total_reward", "one_episode_time", "env_times_all", "model_times_all", "steps_done"]
                excel_path = f"./output/DQN_{task}_result.xlsx"
                df.to_excel(excel_path, index=False)
                print("excel saved!")
                
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 5000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    
    # random seed
    SEED = 525
    random.seed(SEED)
    torch.manual_seed(seed=SEED)
    random.seed(SEED)    

    # create environment
    # env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    task = "PongNoFrameskip"
    env = make_env(env)
    env.action_space.seed(seed=SEED)
    n_actions = env.action_space.n
    n_frames = env.frames.maxlen
    
    print("n_actions:", n_actions, "n_frames:", n_frames)

    # create networks
    policy_net = DQN(in_channels=n_frames, n_actions=n_actions).to(device)
    target_net = DQN(in_channels=n_frames, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    train(env, 100000)
    # torch.save(policy_net, "dqn_pong_model")
    # policy_net = torch.load("dqn_pong_model")
    # test(env, 1, policy_net, render=False)

