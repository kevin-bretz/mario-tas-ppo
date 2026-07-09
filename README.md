# Super Mario World TAS Agent

A reinforcement-learning agent that learns to speedrun the first level of *Super Mario World* (SNES) frame by frame, in the spirit of a tool-assisted speedrun. The PPO implementation, the vectorised environment and the reward are all written from scratch in PyTorch. Solo project for the Game AI course at Leiden.

## How it works

- **Algorithm** — Proximal Policy Optimisation with a clipped surrogate objective, Generalised Advantage Estimation, and a decaying entropy bonus, implemented by hand in `PPO/ppo_agent.py`.
- **Network** — a shared 4-layer CNN (Nature-DQN style) with separate actor and critic heads, reading 6 stacked 84×84 grayscale frames.
- **Environment** — the SNES runs under [stable-retro](https://github.com/Farama-Foundation/stable-retro) with the `SuperMarioWorld-Snes` integration, driven by a custom multi-process vectorised env so many emulator copies step in parallel.
- **Actions** — a curated discrete space of 42 one-to-four-button SNES combos (run / jump / spin-jump directional mixes, plus no-op).
- **Reward** — forward x-progress plus a large bonus for reaching the end of the level; episodes end on death, power-up loss, falling, timeout or completion. I used Cheat Engine to find the SNES RAM addresses behind x-position, lives, power-up state and the end-of-level flag.

## Files

- `train.py` — training loop (vectorised env, PPO update, checkpointing).
- `evaluate.py` — run a trained policy.
- `PPO/ppo_agent.py`, `PPO/ppo_config.py` — the agent and its hyperparameters.
- `plot_training_results.py` — plots return ± std and policy entropy from the `training_results*.json` logs.
- `LVL1_AS2.bk2` — a recorded emulator movie of a run.

## Running

You need the SNES *Super Mario World* ROM (not included — bring your own), imported into stable-retro:

```bash
pip install torch stable-retro opencv-python numpy
python -m retro.import /path/to/your/rom   # register the ROM
python train.py
python evaluate.py
```

## Trained weights

The trained checkpoints (`best_model.pth`, `progress.pth`, ~16 MB each) are hosted on Hugging Face: https://huggingface.co/kevin-bretz/mario-tas-ppo
