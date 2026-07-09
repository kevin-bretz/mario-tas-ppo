import torch
import torch.nn as nn
import numpy as np
import retro
import gym
from gym.spaces import Discrete
from collections import deque
import cv2
import matplotlib.pyplot as plt
import time
import threading

def preprocess_state(rgb_array, new_shape=(84, 84)):
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, new_shape, interpolation=cv2.INTER_AREA)
    return resized[np.newaxis, :, :]

class SimplifiedActionWrapper(gym.Wrapper):
    def __init__(self, local_env):
        super(SimplifiedActionWrapper, self).__init__(local_env)
        self.actions = [
            [0,0,0,0,0,0,0,1,0,0,0,0],  # RIGHT
            [0,0,0,0,0,0,1,0,0,0,0,0],  # LEFT
            [1,0,0,0,0,0,0,0,0,0,0,0],  # B (jump)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # Y (run/fire)
            [0,0,0,0,0,0,0,0,0,1,0,0],  # X (spin jump)
            [0,0,0,0,1,0,0,0,0,0,0,0],  # UP
            [0,0,0,0,0,1,0,0,0,0,0,0],  # DOWN
            [0,1,0,0,0,0,0,1,0,0,0,0],  # Y + RIGHT (run right)
            [1,1,0,0,0,0,0,1,0,0,0,0],  # B + Y (jump while running)
            [1,0,0,0,0,0,0,1,0,0,0,0],  # B + RIGHT (jump right)
            [0,1,0,0,0,0,1,0,0,0,0,0],  # Y + LEFT (run left)
            [1,1,0,0,0,0,1,0,0,0,0,0],  # B + Y + LEFT (jump while running left)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # B + LEFT (jump left)
            [0,0,0,0,1,0,0,1,0,0,0,0],  # UP + RIGHT (climb up/right)
            [0,0,0,0,0,1,0,1,0,0,0,0],  # DOWN + RIGHT (duck/slide right)
            [0,0,0,0,1,0,1,0,0,0,0,0],  # UP + LEFT (climb up/left)
            [0,0,0,0,0,1,1,0,0,0,0,0],  # DOWN + LEFT (duck/slide left)
            [1,0,0,0,0,0,0,0,1,0,0,0],  # B + A (jump + spin jump)
            [0,0,0,0,0,0,0,1,0,1,0,0],  # RIGHT + X (spin jump right)
            [0,0,0,0,0,0,1,0,0,1,0,0],  # LEFT + X (spin jump left)
            [0,0,0,0,0,1,0,0,1,0,0,0],  # DOWN + A (drop through platform)
            [1,1,0,0,0,0,0,1,0,0,0,0],  # B + Y + RIGHT (run jump right)
            [1,1,0,0,0,0,1,0,0,0,0,0],  # B + Y + LEFT (run jump left)
            [1,0,0,0,1,0,0,1,0,0,0,0],  # B + UP + RIGHT (jump up/right)
            [1,0,0,0,0,1,0,1,0,0,0,0],  # B + DOWN + RIGHT (jump down/right)
            [1,0,0,0,1,0,1,0,0,0,0,0],  # B + UP + LEFT (jump up/left)
            [1,0,0,0,0,1,1,0,0,0,0,0],  # B + DOWN + LEFT (jump down/left)
            [1,1,0,0,0,0,0,1,1,0,0,0],  # B + Y + RIGHT + A (run jump + spin)
            [1,1,0,0,0,0,1,0,1,0,0,0],  # B + Y + LEFT + A (run jump left + spin)
            [0,1,0,0,1,0,0,1,0,0,0,0],  # Y + UP + RIGHT (run up/right)
            [0,1,0,0,0,1,0,1,0,0,0,0],  # Y + DOWN + RIGHT (run down/right)
            [0,1,0,0,1,0,1,0,0,0,0,0],  # Y + UP + LEFT (run up/left)
            [0,1,0,0,0,1,1,0,0,0,0,0],  # Y + DOWN + LEFT (run down/left)
            [1,1,0,0,1,0,0,1,0,0,0,0],  # B + Y + UP + RIGHT (run jump up/right)
            [1,1,0,0,0,1,0,1,0,0,0,0],  # B + Y + DOWN + RIGHT (run jump down/right)
            [1,1,0,0,1,0,1,0,0,0,0,0],  # B + Y + UP + LEFT (run jump up/left)
            [1,1,0,0,0,1,1,0,0,0,0,0],  # B + Y + DOWN + LEFT (run jump down/left)
            [1,1,0,0,1,0,0,1,1,0,0,0],  # B + Y + UP + RIGHT + A (run jump up/right + spin)
            [1,1,0,0,0,1,0,1,1,0,0,0],  # B + Y + DOWN + RIGHT + A (run jump down/right + spin)
            [1,1,0,0,1,0,1,0,1,0,0,0],  # B + Y + UP + LEFT + A (run jump up/left + spin)
            [1,1,0,0,0,1,1,0,1,0,0,0],  # B + Y + DOWN + LEFT + A (run jump down/left + spin)
            [0,0,0,0,0,0,0,0,0,0,0,0],  # NOOP
        ]
        self.action_space = Discrete(len(self.actions))

    def step(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return self.env.step(self.actions[action])

class PPONetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(PPONetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *state_size)
            cnn_out = self.cnn(dummy)
            conv_out_size = cnn_out.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return self.policy_head(x), self.value_head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = retro.make(game='SuperMarioWorld-Snes', state='Start')
env = SimplifiedActionWrapper(env)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

NUM_STACK = 6
frame_stack = deque([preprocess_state(obs) for _ in range(NUM_STACK)], maxlen=NUM_STACK)
state_shape = (NUM_STACK, 84, 84)

hidden_size = 512
action_size = env.action_space.n

model = PPONetwork(state_shape, hidden_size, action_size).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))    # - Load the trained model
model.eval()

done = False
step_count = 0
while not done:
    frame_stack.append(preprocess_state(obs))
    obs_stack = np.concatenate(list(frame_stack), axis=0)
    obs_tensor = torch.tensor(obs_stack / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(obs_tensor)


        # action = torch.argmax(policy_logits, dim=1).item()                         # <-- Deterministic action selection

        action_dist = torch.distributions.Categorical(logits=policy_logits)
        action = action_dist.sample().item()                                       # <-- Stochastic action selection

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1/80)
    step_count += 1

env.close()