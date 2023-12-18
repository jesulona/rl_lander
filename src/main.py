# main.py
import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn.dqn import DQNAgent
from dueling_dqn.dueling_dqn import DuelingDQNAgent

# def train_agent(agent, env, episodes):
#     rewards = []
#     for e in range(episodes):
#         state, _ = env.reset(seed=42)
        
#         # Ensure the state is an array and reshape it
#         state = np.array(state).reshape(1, -1)
#         total_reward = 0

#         for time_step in range(500):
#             action = agent.act(state)
#             step = env.step(action)
#             #print(step)
#             next_state, reward, terminated, truncated, _ = step
            
#             # Reshape the next_state similarly
#             next_state = np.array(next_state).reshape(1, -1)

#             agent.remember(state, action, reward, next_state, terminated or truncated)
#             state = next_state
#             total_reward += reward

#             if terminated or truncated:
#                 break

#             agent.replay(32)

#         # Decay epsilon after each episode
#         agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)
#         print(f"Episode: {e}, Epsilon: {agent.epsilon}, Reward: {total_reward}")
#         rewards.append(total_reward)
    
#     return rewards


def train_agent(agent, env, episodes):
    rewards = []
    for e in range(episodes):
        state, _ = env.reset(seed=42)
        state = np.array(state).reshape(1, -1)
        total_reward = 0

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)

            # Use 'step' instead of 'remember' and 'replay'
            agent.step(state, action, reward, next_state, terminated or truncated)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        # Decay epsilon after each episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)
        print(f"Episode: {e}, Epsilon: {agent.epsilon}, Reward: {total_reward}")
        rewards.append(total_reward)

    return rewards

def plot_rewards(rewards_dqn, rewards_dueling_dqn):
    plt.plot(rewards_dqn, label='DQN')
    plt.plot(rewards_dueling_dqn, label='Dueling DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define a seed for reproducibility
    seed = 42

    dqn_agent = DQNAgent(state_size, action_size, seed)
    dueling_dqn_agent = DuelingDQNAgent(state_size, action_size, seed)

    episodes = 10
    #rewards_dqn = train_agent(dqn_agent, env, episodes)
    rewards_dueling_dqn = train_agent(dueling_dqn_agent, env, episodes)

    #plot_rewards(rewards_dqn, rewards_dueling_dqn)

  # Save models
    model_data = [
        ("dqn", dqn_agent.qnetwork),
        ("dueling_dqn", dueling_dqn_agent.qnetwork_local)
    ]

    # Assuming 'model_data' contains tuples of (model_name, model_object)
    # Example: model_data = [('dqn', dqn_agent), ('dueling_dqn', dueling_dqn_agent)]
    for model_name, model in model_data:
        # Construct the path for the model directory
        model_folder = os.path.join('src', model_name, 'models')
        # Create the directory if it does not exist
        os.makedirs(model_folder, exist_ok=True)
        # Construct the full path for the model file
        model_path = os.path.join(model_folder, 'model.pth')
        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")  # Confirmation message

