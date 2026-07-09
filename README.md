# Super Mario World TAS Agent

A reinforcement learning agent that learns to speedrun the first level of Super Mario World (SNES) frame by frame, in the spirit of a tool assisted speedrun. The PPO algorithm, the parallel emulator setup and the reward are all my own code rather than a library. Solo project for the Game AI course at Leiden.

## How it works

The agent reads the game as a stack of 84 by 84 grayscale frames and runs them through a small convolutional network with separate actor and critic heads (`PPO/ppo_agent.py`). It learns with PPO using a clipped surrogate objective, generalised advantage estimation and an entropy bonus, all written by hand. The SNES runs under [stable-retro](https://github.com/Farama-Foundation/stable-retro), and a custom multi process vectorised environment runs many emulator copies in parallel over pipes. The action space is a curated set of 42 button combinations rather than the full controller. The reward is forward x progress plus a large bonus for finishing the level, and an episode ends on death, a fall, a lost power up, a timeout or completion. I used Cheat Engine to find the SNES memory addresses behind Mario's position, lives, power up state and the end of level flag.

## Files

- `train.py` runs the training loop over the vectorised environment.
- `evaluate.py` runs a trained policy.
- `PPO/ppo_agent.py` and `PPO/ppo_config.py` hold the agent and its hyperparameters.
- `plot_training_results.py` plots return and policy entropy from the `training_results*.json` logs.
- `LVL1_AS2.bk2` is a recorded emulator movie of a run.

## Running

You need the SNES Super Mario World ROM (not included, bring your own), imported into stable-retro with `python -m retro.import`. Training runs against the `SuperMarioWorld-Snes` integration.

## Trained weights

The trained checkpoints (`best_model.pth`, `progress.pth`) are on Hugging Face at https://huggingface.co/kevin-bretz/mario-tas-ppo
