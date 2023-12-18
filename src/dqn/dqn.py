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
        self.memory = deque(maxlen=int(1e5))
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=5e-4)

        self.memory = deque(maxlen=int(1e5))
        self.batch_size = 64
        self.gamma = 0.99
        self.update_every = 4
        self.t_step = 0

        #random
        self.epsilon = 0.1  # Initial epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        print("step")
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            print("learned")
            experiences = random.sample(self.memory, k=self.batch_size)
            self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        eps = self.epsilon
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        ran = random.random()
        if ran > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def replay(self, batch_size):
        """Retrieve a batch of experiences from memory and learn from them."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        # Efficient conversion to numpy arrays
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Converting numpy arrays to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Reshape states and next_states to remove the extra dimension
        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        #print("Corrected Shapes - states: {}, next_states: {}".format(states.shape, next_states.shape))

        # Compute Q targets for next states
        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(-1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(-1)))

        # Printing shapes before the line that causes the error
        #print("Corrected Shapes - Q_targets: {}, actions: {}".format(Q_targets.shape, actions.shape))

        # Get expected Q values from local model
        Q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
