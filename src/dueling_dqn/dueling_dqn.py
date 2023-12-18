import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # Value stream
        self.value_stream = nn.Linear(64, 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(64, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DuelingDQNAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Dueling Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-4)

        # Replay memory
        self.memory = deque(maxlen=int(1e5))
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3
        self.update_every = 4
        self.t_step = 0

        #random
        self.epsilon = 0.1  # Initial epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))
        #print("step")
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, k=self.batch_size)
            self.learn(experiences, self.gamma)
            #print("Learning step triggered")  # Print statement to confirm learning


    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""
        eps =self.epsilon
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Debug: Print the state and corresponding action values
        #print("State:", state)
        #print("Action values:", action_values)

        # Epsilon-greedy action selection
        ran = random.random()
        if ran > eps:
            selected_action = np.argmax(action_values.cpu().data.numpy())
            #print(f"Selected action (exploit): {selected_action}")
            return selected_action
        else:
            selected_action = random.choice(np.arange(self.action_size))
            #print(f"Selected action (explore): {selected_action}")
            return selected_action

        # Debug: Print updated epsilon value
        print(f"Updated Epsilon: {self.epsilon}")



    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        # Print shapes of states, actions, rewards, next_states, and dones
        #print(f"Shapes - states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, next_states: {next_states.shape}, dones: {dones.shape}")

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Print shapes of Q_targets_next and Q_targets
        #print(f"Shapes - Q_targets_next: {Q_targets_next.shape}, Q_targets: {Q_targets.shape}")

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Print shape of Q_expected
        #print(f"Shape - Q_expected: {Q_expected.shape}")

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
                    

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)


    def replay(self, batch_size):
        """Retrieve a batch of experiences from memory and learn from them."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Reshape states and next_states to remove the extra dimension
        states = states.squeeze(1)
        next_states = next_states.squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(-1)))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

