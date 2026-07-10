config = {
    "hidden_size": 512,
    "learning_rate": 0.0001,
    "gamma": 0.999,
    "num_episodes": 1000,
    "num_runs": 5,
    "seed": 42,
    "epsilon": 0.2,  # Clipping parameter for PPO
    "entropy_coef": 0.01,  # Static entropy coefficient
    "batch_size": 16392,
    "num_epochs": 4,
    # Reconstructed to match train.py (the matching revision of this file was lost).
    "num_stack": 4,             # grayscale frames stacked as the CNN input channels
    "num_envs": 8,              # parallel emulator copies driven over pipes
    "max_time": 60,             # seconds; the per-episode step cap is max_time * 60
    "entropy_coef_min": 0.001,  # floor the decaying entropy bonus settles at
    "entropy_coef_decay": 0.995 # entropy_coef is multiplied by this each episode
}