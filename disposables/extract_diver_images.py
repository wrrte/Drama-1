import numpy as np
import os
import sys

try:
    from PIL import Image
except ImportError:
    print("PIL not found")
    sys.exit(1)

data = np.load('demonstrations/Seaquest_20260727_230618.npz')
rams = data['ram']
obs = data['obs']

# Diver count is at RAM index 62
diver_counts = rams[:, 62]

out_dir = '/home/ai2lab/.gemini/antigravity-ide/brain/1504effd-2961-409e-97bd-36102558bac4'

for count in range(7):
    # Find indices where diver count equals `count`
    indices = np.where(diver_counts == count)[0]
    
    if len(indices) == 0:
        print(f"No frames found for {count} divers.")
        continue
    
    # Pick up to 2 frames (e.g., first and somewhere in the middle to get variety)
    selected_indices = []
    selected_indices.append(indices[0])
    if len(indices) > 1:
        selected_indices.append(indices[len(indices)//2])
        
    for i, idx in enumerate(selected_indices):
        img_array = obs[idx]
        img = Image.fromarray(img_array)
        img_path = os.path.join(out_dir, f'diver_{count}_img_{i+1}.png')
        img.save(img_path)
        print(f"Saved {img_path}")

