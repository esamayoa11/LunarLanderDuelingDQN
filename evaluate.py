import gymnasium as gym
import torch
import os
from train import DuelingDQN

# FORCE LOAD FROM SAME FOLDER
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "lander.mp4")

env = gym.make("LunarLander-v3", render_mode="rgb_array")
model = DuelingDQN(8, 4)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

frames = []
s, info = env.reset()

for _ in range(1000):
    with torch.no_grad():
        a = model(torch.FloatTensor(s).unsqueeze(0)).argmax().item()
    s, _, term, trunc, _ = env.step(a)

    frames.append(env.render())

    if term or trunc:
        break

# save mp4
import imageio
imageio.mimsave(OUTPUT_VIDEO, frames, fps=30)

env.close()
print(f"Saved video to {OUTPUT_VIDEO}")