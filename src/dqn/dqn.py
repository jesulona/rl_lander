# dqn_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        # Initialize memory (replay buffer)
        #self.memory = deque(maxlen=int(1e5))
        
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, seed, device= "cpu"):
        self.device = device
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=9e-4)

        self.memory = deque(maxlen=int(1e6))
        self.batch_size = 64 *2
        self.gamma = 0.99
        self.update_every = 4
        self.t_step = 0

        #random
        self.epsilon = 1.0  # Initial epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        done = torch.tensor(done).to(self.device)
        self.memory.append((state.cpu().numpy(), action.cpu().item(), reward.cpu().item(), next_state.cpu().numpy(), done.cpu().item()))
        #self.memory.append((state, action, reward, next_state, done))
        #print("step")
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            #print("learned")
            experiences = random.sample(self.memory, k=self.batch_size)
            self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        eps = self.epsilon
        #state = torch.from_numpy(state).float().unsqueeze(0)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        ran = random.random()
        if ran > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # def learn(self, experiences, gamma):
    #     states, actions, rewards, next_states, dones = zip(*experiences)

    #     states = torch.from_numpy(np.vstack(states)).float()
    #     actions = torch.from_numpy(np.vstack(actions)).long()
    #     rewards = torch.from_numpy(np.vstack(rewards)).float()
    #     next_states = torch.from_numpy(np.vstack(next_states)).float()
    #     dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

    #     Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    #     Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    #     Q_expected = self.qnetwork(states).gather(1, actions)

    #     #loss = F.mse_loss(Q_expected, Q_targets)
    #     loss = F.smooth_l1_loss(Q_expected, Q_targets)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), 1.0)  # 1.0 is the clip value, can be adjusted
    #     self.optimizer.step()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), 5.0)
        self.optimizer.step()


    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

  