import multiprocessing as mp
import numpy as np
import retro
from gym.spaces import Discrete
from gym import Wrapper
from PPO.ppo_agent import PPOAgent
from PPO.ppo_config import config
import os
import torch
import glob
import shutil
import subprocess
import cv2
from collections import deque
import re
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results_log = []
results_json_path = "training_results.json"

class SimplifiedActionWrapper(Wrapper):
    def __init__(self, local_env):
        super(SimplifiedActionWrapper, self).__init__(local_env)
        self.actions = [
            # 1-button actions
            [0,0,0,0,0,0,0,1,0,0,0,0],  # RIGHT
            [0,0,0,0,0,0,1,0,0,0,0,0],  # LEFT
            [1,0,0,0,0,0,0,0,0,0,0,0],  # B (jump)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # Y (run/fire)
            [0,0,0,0,0,0,0,0,0,1,0,0],  # X (spin jump)
            [0,0,0,0,1,0,0,0,0,0,0,0],  # UP
            [0,0,0,0,0,1,0,0,0,0,0,0],  # DOWN
            # 2-button combos
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
            # 3-button combos
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
            # 4-button combos (for advanced moves)
            [1,1,0,0,1,0,0,1,0,0,0,0],  # B + Y + UP + RIGHT (run jump up/right)
            [1,1,0,0,0,1,0,1,0,0,0,0],  # B + Y + DOWN + RIGHT (run jump down/right)
            [1,1,0,0,1,0,1,0,0,0,0,0],  # B + Y + UP + LEFT (run jump up/left)
            [1,1,0,0,0,1,1,0,0,0,0,0],  # B + Y + DOWN + LEFT (run jump down/left)
            [1,1,0,0,1,0,0,1,1,0,0,0],  # B + Y + UP + RIGHT + A (run jump up/right + spin)
            [1,1,0,0,0,1,0,1,1,0,0,0],  # B + Y + DOWN + RIGHT + A (run jump down/right + spin)
            [1,1,0,0,1,0,1,0,1,0,0,0],  # B + Y + UP + LEFT + A (run jump up/left + spin)
            [1,1,0,0,0,1,1,0,1,0,0,0],  # B + Y + DOWN + LEFT + A (run jump down/left + spin)
            # NOOP
            [0,0,0,0,0,0,0,0,0,0,0,0],  # NOOP
        ]
        self.action_space = Discrete(len(self.actions))

    def step(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return self.env.step(self.actions[action])


def custom_reward_function(prev_info, curr_info, max_x_reached, step_count, timed_out=False, delta_x_running_max=1.0):
    reward = 0
    x_prev = prev_info.get("x", 16)
    x_curr = curr_info.get("x", 16)
    delta_x = x_curr - x_prev

    bonus = 0.01
    if x_curr > max_x_reached:
        reward += ((x_curr - max_x_reached) * bonus)
        max_x_reached = x_curr

    # Penalty for losing powerup
    # if curr_info.get("powerup", 0) < prev_info.get("powerup", 0):
    #     reward -= 50

    # Big reward for finishing
    if curr_info.get("endOfLevel", False):
        reward += 50
    
    # reward -= 0.01

    # Penalty for losing a life
    # if curr_info.get("lives", 1) < prev_info.get("lives", 1):
    #     reward -= 50
    
    # Penalty for falling
    # if abs(x_curr - x_prev) > 100:
    #     reward -= 50

    # Penalty for timeout
    # if timed_out:
    #     reward -= 7.2

    return reward, max_x_reached


def worker_process(worker_id, conn, game, state, config, record_path=None):
    num_stack = config["num_stack"]
    env = retro.make(game=game, state=state, render_mode="rgb_array", use_restricted_actions=retro.Actions.ALL, record=record_path)
    env = SimplifiedActionWrapper(env)
    state, info = env.reset()
    state = preprocess_state(state, new_shape=(84, 84))
    frame_stack = deque([state.squeeze(0) for _ in range(num_stack)], maxlen=num_stack)
    conn.send((np.stack(frame_stack, axis=0), 0, False, info))
    prev_info = info

    delta_x_running_max = 1.0
    max_steps = config["max_time"] * 60
    step_count = 0
    done = False
    last_state = np.stack(frame_stack, axis=0)
    last_reward = 0
    last_info = info
    max_x_reached = 16

    while True:
        action = conn.recv()

        if action == "reset":
            state, info = env.reset()
            state = preprocess_state(state, new_shape=(84, 84))
            frame_stack = deque([state.squeeze(0) for _ in range(num_stack)], maxlen=num_stack)
            conn.send((np.stack(frame_stack, axis=0), 0, False, info))
            prev_info = info
            step_count = 0
            done = False
            last_state = np.stack(frame_stack, axis=0)
            last_reward = 0
            last_info = info
            max_x_reached = 0
            continue

        if done:
            conn.send((last_state, last_reward, True, last_info))
            continue

        next_state, raw_reward, term, trunc, curr_info = env.step(action)
        next_state = preprocess_state(next_state, new_shape=(84, 84))
        frame_stack.append(next_state.squeeze(0))
        step_count += 1

        timed_out = step_count >= max_steps

        reward, max_x_reached = custom_reward_function(
            prev_info, curr_info, max_x_reached, step_count, timed_out=timed_out, delta_x_running_max=delta_x_running_max
        )

        x_prev = prev_info.get("x", 16)
        x_curr = curr_info.get("x", 16)
        fell = abs(x_prev - x_curr) > 100

        done = (
            term or
            trunc or
            (curr_info.get("lives", 1) < prev_info.get("lives", 1)) or
            (curr_info.get("powerup", 0) < prev_info.get("powerup", 0)) or
            curr_info.get("endOfLevel", False) or
            timed_out or
            fell
        )

        done_reason = None
        if curr_info.get("endOfLevel", False):
            done_reason = "endOfLevel"
        elif fell:
            done_reason = "fell"
        elif curr_info.get("lives", 1) < prev_info.get("lives", 1):
            done_reason = "lost_life"
        elif curr_info.get("powerup", 0) < prev_info.get("powerup", 0):
            done_reason = "lost_powerup"
        elif timed_out:
            done_reason = "timeout"
        elif term or trunc:
            done_reason = "terminated"

        curr_info["done_reason"] = done_reason

        stacked_state = np.stack(frame_stack, axis=0)
        assert stacked_state.shape == (num_stack, 84, 84), f"State shape is {stacked_state.shape}, expected ({num_stack}, 84, 84)"
        conn.send((stacked_state, reward, done, curr_info))

        prev_info = curr_info

        if done:
            last_state = stacked_state
            last_reward = reward
            last_info = curr_info


class VectorizedEnv:
    def __init__(self, num_envs, game, state, config, record_paths=None):
        self.num_envs = num_envs
        self.game = game
        self.state = state
        self.config = config
        self.processes = []
        self.parent_conns = []
        self.child_conns = []

        for i in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            rec_path = record_paths[i] if record_paths else None
            process = mp.Process(target=worker_process, args=(i, child_conn, game, state, config, rec_path))
            process.start()
            self.processes.append(process)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def step(self, actions):
        for conn, action in zip(self.parent_conns, actions):
            conn.send(action)
        results = [conn.recv() for conn in self.parent_conns]
        next_states, rewards, dones, infos = zip(*results)
        for idx, s in enumerate(next_states):
            assert s.shape == next_states[0].shape, f"Shape mismatch at env {idx}: {s.shape} vs {next_states[0].shape}"
        return np.stack(next_states), np.array(rewards), np.array(dones), infos

    def close(self):
        for process in self.processes:
            process.terminate()
        for conn in self.child_conns:
            conn.close()


def preprocess_state(rgb_array, new_shape=(84, 84)):
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, new_shape, interpolation=cv2.INTER_AREA)
    return resized[np.newaxis, :, :]


start_episode = 0

all_time_best_reward = float('-inf')
last_best_episode = start_episode
best_model_path = os.path.join("recordings", "best_model.pth")
recordings_dir = "recordings"
os.makedirs(recordings_dir, exist_ok=True)

num_envs = config["num_envs"]
vectorized_env = VectorizedEnv(num_envs, "SuperMarioWorld-Snes", "Start", config)

state_dim = (config["num_stack"], 84, 84)
dummy_env = retro.make(game="SuperMarioWorld-Snes", state="Start")
action_dim = len(SimplifiedActionWrapper(dummy_env).actions)
dummy_env.close()
ppo_agent = PPOAgent(
    env=None,
    state_size=state_dim,
    hidden_size=config["hidden_size"],
    action_size=action_dim,
    learning_rate=config["learning_rate"],
    gamma=config["gamma"],
    epsilon=config["epsilon"],
    entropy_coef=config["entropy_coef"],
    batch_size=config["batch_size"],
    num_epochs=config["num_epochs"],
    num_stack=config["num_stack"],
    device=device,
    entropy_coef_min=config["entropy_coef_min"],
    entropy_coef_decay=config["entropy_coef_decay"]
)

model_path = "recordings/BASE/model_ep300.pth"
if os.path.exists(model_path):
    ppo_agent.network.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
else:
    print("No model loaded, starting from scratch.")

max_steps_per_episode = config["max_time"] * 60 * config["num_envs"]
num_episodes = 100000

for episode in range(start_episode, start_episode + num_episodes):
    print(f"Starting Episode {episode + 1}...")

    ppo_agent.decay_entropy_coef()
    print(f"Entropy coefficient: {ppo_agent.entropy_coef:.8f}")

    vectorized_env.close()
    vectorized_env = VectorizedEnv(
        num_envs,
        "SuperMarioWorld-Snes",
        "Start",
        config,
        record_paths=None
    )

    initial_data = [conn.recv() for conn in vectorized_env.parent_conns]
    states = [data[0] for data in initial_data]
    infos = [data[3] for data in initial_data]
    done = [False] * num_envs
    episode_rewards = [0] * num_envs
    episode_steps = [0] * num_envs
    steps = 0

    while not all(done) and steps < max_steps_per_episode:
        actions = []
        for i in range(num_envs):
            if done[i]:
                actions.append(0)
            else:
                action, _, _ = ppo_agent.select_action(states[i])
                actions.append(action)

        next_states, rewards, dones, infos = vectorized_env.step(actions)

        for i in range(num_envs):
            if not done[i]:
                ppo_agent.add_experience(states[i], actions[i], rewards[i], next_states[i], dones[i])
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                done[i] = dones[i]
                states[i] = next_states[i]

        steps += 1

        if len(ppo_agent.states) >= config["batch_size"]:
            ppo_agent.update()

    for env_idx in range(num_envs):
        x = infos[env_idx].get('x', 0)
        total_reward = episode_rewards[env_idx]
        steps_taken = episode_steps[env_idx]
        done_reason = infos[env_idx].get("done_reason", "unknown")
        if done_reason == "endOfLevel":
            reason_str = "🏁 end of level"
        elif done_reason == "timeout":
            reason_str = "⏰ timed out"
        elif done_reason == "lost_life":
            reason_str = "💀 death"
        elif done_reason == "fell":
            reason_str = "💀 fell"
        elif done_reason == "lost_powerup":
            reason_str = "🪙 lost powerup"
        elif done_reason == "terminated":
            reason_str = "🛑 terminated"
        else:
            reason_str = f"other ({done_reason})"
        print(f"Episode {episode + 1} - Env {env_idx}: final x={x}, total reward={total_reward:.3f}, steps={steps_taken} ({reason_str})")
        if total_reward > all_time_best_reward:
            all_time_best_reward = total_reward

    print(f"All-time highest reward so far: {all_time_best_reward:.3f}")

    if (episode + 1) % 10 == 0:
        model_save_path = f"recordings/model_ep{episode+1}.pth"
        torch.save(ppo_agent.network.state_dict(), model_save_path)
        print(f"Model checkpoint saved: {model_save_path}")

    with torch.no_grad():
        all_states = np.array(ppo_agent.states)
        if len(all_states) > 0:
            states_t = torch.tensor(all_states, dtype=torch.float32).to(device) / 255.0
            logits, _ = ppo_agent.network(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().mean().item()
            print(f"Mean policy entropy over episode: {entropy:.3f}")
        else:
            print("No states collected for entropy calculation.")

    episode_returns = np.array(episode_rewards)
    avg_return = episode_returns.mean()
    std_return = episode_returns.std()
    avg_entropy = entropy if 'entropy' in locals() else None

    results_log.append({
        "episode": episode + 1,
        "avg_return": float(avg_return),
        "std_return": float(std_return),
        "avg_entropy": float(avg_entropy) if avg_entropy is not None else None
    })

    if (episode + 1) % 10 == 0:
        with open(results_json_path, "w") as f:
            json.dump(results_log, f, indent=2)