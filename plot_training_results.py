import json
import matplotlib.pyplot as plt

with open("training_results_combined.json", "r") as f:
    results = json.load(f)

episodes = [entry["episode"] for entry in results]
avg_returns = [entry["avg_return"] for entry in results]
std_returns = [entry["std_return"] for entry in results]
avg_entropy = [entry["avg_entropy"] for entry in results]

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(episodes, avg_returns, label="Average Return")
plt.fill_between(episodes,
                 [a - s for a, s in zip(avg_returns, std_returns)],
                 [a + s for a, s in zip(avg_returns, std_returns)],
                 color='blue', alpha=0.2, label="Return ± Std")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.title("Training Performance")

plt.subplot(2, 1, 2)
plt.plot(episodes, avg_entropy, color='orange', label="Average Entropy")
plt.xlabel("Episode")
plt.ylabel("Entropy")
plt.legend()
plt.tight_layout()
plt.show()