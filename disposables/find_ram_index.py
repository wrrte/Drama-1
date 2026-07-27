import numpy as np

try:
    data = np.load('demonstrations/Seaquest_20260727_230618.npz')
except Exception as e:
    print(f"Error loading npz: {e}")
    exit(1)

rams = data['ram']
obs = data['obs']

print("Total frames:", len(rams))

# Find indices that take values from 0 to 6 (or subset) and usually increase by 1
possible_indices = []
for i in range(128):
    unique_vals = np.unique(rams[:, i])
    # Divers start at 0, max 6
    if np.all(np.isin(unique_vals, [0, 1, 2, 3, 4, 5, 6])):
        # Check if they step up
        diffs = np.diff(rams[:, i].astype(int))
        if np.any(diffs == 1):
            possible_indices.append((i, unique_vals))

print("Possible RAM indices for diver count:")
for idx, vals in possible_indices:
    print(f"Index: {idx}, Unique values: {vals}")
