import json

# Load both files
with open('training_results0.json', 'r') as f:
    results0 = json.load(f)
with open('training_results.json', 'r') as f:
    results = json.load(f)

# Find the last episode number in results0
last_episode = results0[-1]['episode'] if results0 else 0

# Increment episode numbers in results
for entry in results:
    entry['episode'] += last_episode

# Append and save
combined = results0 + results

with open('training_results_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)