### Overview
This project implements a **Dueling Deep Q-Network (Dueling DQN)** to solve the [OpenAI Gym LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) 
environment. The agent learns to land a rocket safely between the landing pads while maximizing cumulative reward.  

The Dueling DQN architecture separates the estimation of state value from the advantage of each action, improving stability and learning efficiency compared to standard DQNs.
This allows the agent to more accurately evaluate actions in states where some moves are clearly better than others. The project demonstrates both theoretical and practical
aspects of reinforcement learning in a continuous state space with discrete actions.

### Features
- Dueling DQN architecture for improved value estimation.
- Training using PyTorch with experience replay and target network.
- Reproducible results via **seeded environment resets**.

### Requirements
- Python ≥ 3.8
- [gymnasium](https://pypi.org/project/gymnasium/)
- torch
- imageio

Install dependencies: pip install -r requirements.txt

**Train the agent:** python train.py

**Evaluate & record best performance:** evaluate.py

Project Structure

LunarLanderDuelingDQN/
├── evaluate.py       # Optional evaluation script
├── lander.mp4        # Optional recorded gameplay
├── model.pth         # Saved trained model
├── test_lander.py    # Script to test the agent
├── train.py          # Training script for the Dueling DQN
├── __pycache__/      # Compiled Python files
│   └── train.cpython-310.pyc
└── README.md         # Project overview

