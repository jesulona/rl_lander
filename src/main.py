# main.py
import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn.dqn import DQNAgent
from dueling_dqn.dueling_dqn import DuelingDQNAgent
from ddqn.ddqn import DDQNAgent
from ppo.ppo import PPOAgent

# def train_agent(agent, env, episodes, update_timestep=2000):
#     rewards = []
#     timestep = 0

#     for e in range(episodes):
#         state, _ = env.reset(seed=42)
#         state = np.array(state).reshape(1, -1)
#         total_reward = 0

#         for time_step in range(500):
#             if isinstance(agent, PPOAgent):
#                 action, log_prob = agent.select_action(state)
#                 next_state, reward, terminated, truncated, _ = env.step(action)
#                 next_state = np.array(next_state).reshape(1, -1)
#                 agent.step(state, action, log_prob, reward, terminated or truncated)
#             else:
#                 action = agent.act(state)
#                 next_state, reward, terminated, truncated, _ = env.step(action)
#                 next_state = np.array(next_state).reshape(1, -1)
#                 agent.step(state, action, reward, next_state, terminated or truncated)

#             state = next_state
#             total_reward += reward
#             timestep += 1

#             if terminated or truncated:
#                 break

#             if isinstance(agent, PPOAgent) and timestep % update_timestep == 0:
#                 agent.update()

#         if not isinstance(agent, PPOAgent):
#             agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)

#         print(f"Episode: {e}, Reward: {total_reward}")
#         rewards.append(total_reward)

    # return rewards
def train_agent(agent, env, episodes):
    rewards = []
    for e in range(episodes):
        state, _ = env.reset(seed=40)
        state = np.array(state).reshape(1, -1)
        total_reward = 0
        for time_step in range(1000):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
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


def plot_rewards(rewards_dqn, rewards_ddqn):
    plt.plot(rewards_dqn, label='DQN')
    plt.plot(rewards_ddqn, label='DDQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode=None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define a seed for reproducibility
    seed = 40


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dqn_agent = DQNAgent(state_size, action_size, seed, device)

    dqn_agent = DQNAgent(state_size, action_size, seed)
    dueling_dqn_agent = DuelingDQNAgent(state_size, action_size, seed)
    #ddqn_agent = DDQNAgent(state_size,action_size,seed)
    ddqn_agent = DDQNAgent(state_size,action_size,seed,device)
    ppo_agent = PPOAgent(state_size, action_size)



    episodes = 1000
    rewards_dqn = train_agent(dqn_agent, env, episodes)
    #rewards_dueling_dqn = train_agent(dueling_dqn_agent, env, episodes)
    rewards_ddqn = train_agent(ddqn_agent, env, episodes)
    #rewards_ppo = train_agent(ppo_agent, env, episodes)


    plot_rewards(rewards_dqn, rewards_ddqn)
    # plt.plot(rewards_dqn, label='dqn')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()


  # Save models
    model_data = [
        ("dqn", dqn_agent.qnetwork),
        ("ddqn", ddqn_agent.qnetwork_local)
        #("dueling_dqn", dueling_dqn_agent.qnetwork_local)
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

