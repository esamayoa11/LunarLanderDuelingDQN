import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

# SAVE/LOAD TO EXACT SAME FOLDER AS THIS FILE 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# Dueling Q-Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        shared = self.shared(x)
        value = self.value(shared)
        advantage = self.advantage(shared)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Replay Buffer 
class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(ns),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)

# Environment Setup 
env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DuelingDQN(state_dim, action_dim)
target_net = DuelingDQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

episodes = 600
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

# Action Selection 
def select_action(state):
    global epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax().item()

# Training Loop 
for ep in range(episodes):
    s, info = env.reset()
    ep_reward = 0

    for _ in range(1000):
        a = select_action(s)
        ns, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        buffer.add((s, a, r, ns, done))
        s = ns
        ep_reward += r

        if done:
            break

        # Train each step
        if len(buffer) > batch_size:
            S, A, R, NS, D = buffer.sample(batch_size)

            with torch.no_grad():
                next_actions = q_net(NS).argmax(dim=1)
                target_values = target_net(NS).gather(1, next_actions.unsqueeze(1)).squeeze()
                targets = R + gamma * target_values * (1 - D)

            q_values = q_net(S).gather(1, A.unsqueeze(1)).squeeze()
            loss = nn.MSELoss()(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Soft target update
    if ep % 10 == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {ep} | Reward: {ep_reward:.1f} | Epsilon: {epsilon:.3f}")

# SAVE MODEL
torch.save(q_net.state_dict(), MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")

env.close()