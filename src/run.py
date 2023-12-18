import os
import pygame
import gymnasium as gym
import torch
import numpy as np
from dqn.dqn import DQNAgent
from dueling_dqn.dueling_dqn import DuelingDQNAgent

def initialize_pygame():
    pygame.init()
    pygame.display.set_mode((400, 300))

def quit_pygame():
    pygame.display.quit()
    pygame.quit()

def load_model(model_path, agent_class, state_size, action_size, seed):
    # Initialize the agent
    agent = agent_class(state_size, action_size, seed)

    # Determine the correct Q-network attribute based on the agent class
    if hasattr(agent, 'qnetwork'):
        network = agent.qnetwork
    elif hasattr(agent, 'qnetwork_local'):
        network = agent.qnetwork_local
    else:
        raise ValueError("Agent class does not have a recognized Q-network attribute.")

    # Load the model weights
    network.load_state_dict(torch.load(model_path))
    network.eval()

    return agent


def run_episodes(env, agent, n_episodes):
    total_rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset(seed=42)
        total_reward = 0
        while True:
            action = agent.act(state, eps=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        total_rewards.append(total_reward)
    average_reward = sum(total_rewards) / n_episodes
    return average_reward

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    state_size = 8
    action_size = 4
    seed = 42
    n_episodes = 1

    initialize_pygame()

    # Load and run DQN agent
    dqn_model_path = "src/dqn/models/model.pth"
    dqn_agent = load_model(dqn_model_path, DQNAgent, state_size, action_size, seed)
    env = gym.make(env_name, render_mode="human")
    print("Running DQN agent...")
    average_reward_dqn = run_episodes(env, dqn_agent, n_episodes)
    print(f"Average Reward for DQN: {average_reward_dqn}")
    env.close()

    quit_pygame()
    initialize_pygame()

    # Load and run Dueling DQN agent
    dueling_dqn_model_path = "src/dueling_dqn/models/model.pth"
    dueling_dqn_agent = load_model(dueling_dqn_model_path, DuelingDQNAgent, state_size, action_size, seed)
    env = gym.make(env_name, render_mode="human")
    print("Running Dueling DQN agent...")
    average_reward_dueling_dqn = run_episodes(env, dueling_dqn_agent, n_episodes)
    print(f"Average Reward for Dueling DQN: {average_reward_dueling_dqn}")
    env.close()

    quit_pygame()