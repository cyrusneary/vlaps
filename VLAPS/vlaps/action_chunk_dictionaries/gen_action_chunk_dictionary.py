import numpy as np
import pickle
import pathlib

from pathlib import Path

import sys
from vlaps.action_chunk_dictionaries.read_libero_results import read_libero_results
from sklearn_extra.cluster import KMedoids

# Experiment runs from which to grab action chunks.
experiment_names = [
    "2025-07-14_17-33-26_octo_object_ckpt100000",
    "2025-07-14_17-49-27_octo_spatial_ckpt100000",
    "2025-07-14_18-10-01_octo_goal_ckpt100000",
    "2025-07-14_18-51-16_octo_90_ckpt100000",
    "2025-07-14_19-00-59_octo_10_ckpt100000",
    # "2025-06-20_11-33-09_octo_agent_spatial", 
    # "2025-06-20_11-59-16_octo_agent_object", 
    # "2025-06-20_12-39-20_octo_agent_goal"
]

action_chunks = []
subfolders = []

for experiment_name in experiment_names:
    folder = pathlib.Path("/home/mila/n/nearyc/scratch/projects/vlaps_data/experiments/libero/results") / experiment_name
    subfolders.extend(list(folder.iterdir()))

# Load all the data from the subfolders
num_runs = 0
for subfolder in subfolders:
    if str(subfolder)[-7::] == "success" and any(f"init_state_{i}" in str(subfolder) for i in range(1)):# or str(subfolder)[-7::] == "failure":
        num_runs += 1
        file = subfolder / "all_sampled_chunks.pkl"
        data = read_libero_results(file)
        action_chunks = action_chunks + data

# Step 1: Stack all action chunks into one big array
# Shape: (n_chunks, T, d)
actions_array = np.stack(action_chunks)

# save the actions array to a pickle file
with open(f"2025-07-14_octo_all_task_suites_successes_only.pkl", 'wb') as f:
    pickle.dump(actions_array, f)

print(actions_array.shape[0])

# Step 2: Normalize per action dimension
# We want to scale each *action dimension* independently across all timesteps and all chunks

# Reshape to (n_chunks * T, d) to treat time as part of the dataset
actions_2d = actions_array.reshape(-1, actions_array.shape[-1])

# Compute per-dimension 1st and 99th percentiles
low = np.percentile(actions_2d, 1, axis=0)   # (d,)
high = np.percentile(actions_2d, 99, axis=0) # (d,)

# Normalize: for each dimension, map 1st percentile to -1 and 99th to +1
actions_norm = (actions_array - low) / (high - low)
actions_norm = actions_norm * 2 - 1  
actions_norm = np.clip(actions_norm, -1, 1)  # Clip outliers

# Step 3: Flatten each normalized action chunk
# From shape (T, d) --> (T*d,)
actions_flat = actions_norm.reshape(actions_norm.shape[0], -1)  # (n_chunks, T*d)

# Step 4: Perform K-Medoids clustering
num_clusters = 2000  # <-- How many prototypes you want in your dictionary

kmedoids = KMedoids(
    n_clusters=num_clusters, 
    random_state=0, 
    metric='euclidean', 
    method='alternate',
    init='k-medoids++'
)
kmedoids.fit(actions_flat)

# Step 5: Extract your dictionary
# These are the indices of the chosen medoids (real action chunks)
dictionary_indices = kmedoids.medoid_indices_

# Your action dictionary (normalized and flattened)
action_dictionary_flat = actions_flat[dictionary_indices]

# If you want to reshape them back to (T, d) sequences:
T, d = action_chunks[0].shape
action_dictionary = action_dictionary_flat.reshape(num_clusters, T, d)

print(action_dictionary)

# save the dictionary to a pickle file
with open(f"2025-07-14_octo_all_task_suites_successes_only_{num_clusters}_medoids.pkl", 'wb') as f:
    pickle.dump(action_dictionary, f)