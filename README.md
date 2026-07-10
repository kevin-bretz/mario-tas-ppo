# Super Mario World TAS Agent

Part of a group project (five of us) for the Modern Game AI course at Leiden, comparing three reinforcement learning algorithms (DQN, PPO and SAC) on speedrunning the first level of Super Mario World (SNES), in the spirit of a tool assisted speedrun. This repository holds the from-scratch PPO agent, with the algorithm, the parallel emulator setup and the reward all written by hand rather than pulled from a library.

## How it works

The agent reads the game as a stack of 84 by 84 grayscale frames and runs them through a small convolutional network with separate actor and critic heads (`PPO/ppo_agent.py`). It learns with PPO using a clipped surrogate objective, generalised advantage estimation and an entropy bonus. The SNES runs under [stable-retro](https://github.com/Farama-Foundation/stable-retro), and a custom multi process vectorised environment runs many emulator copies in parallel over pipes. The action space is a curated set of 42 button combinations rather than the full controller. The reward is forward x progress plus a large bonus for finishing the level, and an episode ends on death, a fall, a timeout or completion. The SNES memory addresses behind Mario's position, lives and the end of level flag were found with the stable-retro integration tool and live in the integration `data.json` (see Running).

## Files

- `train.py` runs the training loop over the vectorised environment.
- `evaluate.py` runs a trained policy.
- `PPO/ppo_agent.py` and `PPO/ppo_config.py` hold the agent and its hyperparameters.
- `plot_training_results.py` plots return and policy entropy from the `training_results*.json` logs.
- `LVL1_AS2.bk2` is a recorded emulator movie of a run.
- `integration/SuperMarioWorld-Snes/data.json` is the custom stable-retro integration that exposes Mario's position, lives and the end of level flag to the reward.

## Running

You need the SNES Super Mario World ROM. It is copyrighted by Nintendo, so it is not included here and cannot be, you have to supply your own legally obtained copy. Put it in a folder and run `python -m retro.import path/to/that/folder`, which recognises the ROM by its checksum and installs it into stable-retro under the `SuperMarioWorld-Snes` integration. Training then runs against that integration.

The reward reads memory values that the stock `SuperMarioWorld-Snes` integration does not expose, so copy the custom integration over the one that ships with stable-retro before training:

```bash
python -c "import retro, os, shutil; shutil.copy('integration/SuperMarioWorld-Snes/data.json', os.path.join(os.path.dirname(retro.__file__), 'data', 'stable', 'SuperMarioWorld-Snes', 'data.json'))"
```

## Trained weights

The trained checkpoints (`best_model.pth`, `progress.pth`) are on Hugging Face at https://huggingface.co/kevin-bretz/mario-tas-ppo
