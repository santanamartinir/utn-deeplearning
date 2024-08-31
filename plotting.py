import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_process_data(filename):
    """ Load data from JSON file and compute average metrics by number of trials. """
    with open(filename, 'r') as file:
        data = json.load(file)

    trial_to_avg_rank = {}
    trial_to_avg_regret = {}

    for experiment_id, tests in data.items():
        for test_id, results in tests.items():
            if isinstance(results, dict):
                results = list(results.values())  # Convert dictionary values to list
            num_trials = len(results)
            avg_rank = np.mean(results)
            avg_regret = np.mean(results)  # Assuming regret is the same as rank here; adjust as necessary
            if num_trials not in trial_to_avg_rank:
                trial_to_avg_rank[num_trials] = []
                trial_to_avg_regret[num_trials] = []
            trial_to_avg_rank[num_trials].append(avg_rank)
            trial_to_avg_regret[num_trials].append(avg_regret)

    num_trials_list = sorted(trial_to_avg_rank.keys())
    avg_rank_list = [np.mean(trial_to_avg_rank[n]) for n in num_trials_list]
    avg_regret_list = [np.mean(trial_to_avg_regret[n]) for n in num_trials_list]

    return num_trials_list, avg_rank_list, avg_regret_list

# File paths
files = ['results/MyAlgorithm.json','results/RandomSearch.json', 'results/GP.json', 'results/DGP.json']
colors = ['r', 'b', 'g', 'y']
labels = ['our DDPM', 'Random Search', 'GP', 'DGP']

# Ensure plots/ directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Create plots
plt.figure(figsize=(14, 10))

# Plot for Average Rank
plt.subplot(2, 1, 1)
for file, color, label in zip(files, colors, labels):
    num_trials_list, avg_rank_list, _ = load_and_process_data(file)
    plt.plot(num_trials_list, avg_rank_list, marker='o', linestyle='-', color=color, label=label)

plt.title('Average Rank vs. Number of Trials')
plt.xlabel('Number of Trials')
plt.ylabel('Average Rank')
plt.legend()
plt.grid(True)

# Plot for Average Regret
plt.subplot(2, 1, 2)
for file, color, label in zip(files, colors, labels):
    num_trials_list, _, avg_regret_list = load_and_process_data(file)
    plt.plot(num_trials_list, avg_regret_list, marker='o', linestyle='-', color=color, label=label)

plt.title('Average Regret vs. Number of Trials')
plt.xlabel('Number of Trials')
plt.ylabel('Average Regret')
plt.legend()
plt.grid(True)

# Save plots
plot_filename = 'plots/average_metrics.png'
plt.tight_layout()
plt.savefig(plot_filename)

# Show plot (optional)
plt.show()
