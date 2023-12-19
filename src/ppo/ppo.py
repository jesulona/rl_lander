import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = self.softmax(self.fc2(x))
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        state_value = self.fc2(x)
        return state_value

class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=0.00003, lr_critic=0.0001, gamma=0.99, eps_clip=0.2, update_timestep=2000):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_timestep = update_timestep
        self.timestep = 0

        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.policy_old = PolicyNetwork(state_size, action_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value_network = ValueNetwork(state_size)
        self.optimizer_critic = optim.Adam(self.value_network.parameters(), lr=lr_critic)

        self.memory = []  # Store (state, action, log_prob, reward, done)

    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1))
    #     with torch.no_grad():
    #         action_probs = self.policy_old(state)
    #     distribution = Categorical(action_probs)
    #     action = distribution.sample()
    #     self.memory.append((state, action, distribution.log_prob(action)))
    #     return action.item()
    
    # def select_action(self, state):
    #     """
    #     Selects an action based on the current policy and returns the action and its log probability.
    #     """
    #     # Convert state to tensor
    #     state = torch.from_numpy(state).float().unsqueeze(0)

    #     # Forward pass through the network
    #     action_probs = self.policy(state)
    #     m = torch.distributions.Categorical(action_probs)

    #     # Sample an action and get its log probability
    #     action = m.sample()
    #     log_prob = m.log_prob(action)

    #     # Convert action to numpy and return it along with log probability
    #     return action.item(), log_prob
    
    def select_action(self, state):
            state = torch.FloatTensor(state.reshape(1, -1))
            with torch.no_grad():
                action_probs = self.policy_old(state)
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            return action.item(), log_prob
    
    def step(self, state, action, log_prob, reward, done):
        # Convert state to a Tensor if it's not already
        state = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state
        self.memory.append((state, action, log_prob, reward, done))


    def update(self):
        rewards = []
        discounted_reward = 0
        for (_, _, _, reward, done) in reversed(self.memory):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor and ensure correct data type
        old_states = torch.stack([s for (s, _, _, _, _) in self.memory]).detach()
        old_actions = torch.tensor([a for (_, a, _, _, _) in self.memory], dtype=torch.int64).unsqueeze(1)

        # Ensure rewards tensor is float32
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(4):
            # Evaluating old actions and values
            action_probs = self.policy(old_states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(old_actions.squeeze())
            state_values = self.value_network(old_states).squeeze()
            advantages = rewards - state_values.detach()

            # Compute loss
            ratios = torch.exp(log_probs - torch.stack([lp for (_, _, lp, _, _) in self.memory]).detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = nn.MSELoss()(state_values, rewards)

            # Update actor
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            # Update critic
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()


        # Clear memory
        self.memory.clear()
        self.policy_old.load_state_dict(self.policy.state_dict())
