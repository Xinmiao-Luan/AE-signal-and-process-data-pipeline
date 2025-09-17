import os
import numpy as np
import pandas as pd

csv_dir = "/Users/xluan3/Desktop/Projects/AE_data/contour_AEdata_71to92"
out_dir = "/Users/xluan3/Desktop/Projects/AE_data/contour_AEdata_71to92_npy"
os.makedirs(out_dir, exist_ok=True)

TARGET_HEIGHT = 128
TARGET_WIDTH = 128
TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH

for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(csv_path, header=None).values  # shape: [T, F]

        # Option 1: use just one feature column (e.g., column 0)
        if data.shape[1] > 1:
            data = data[:, 0]

        # Flatten
        flat = data.flatten()

        # Pad or crop to match size
        if flat.shape[0] > TARGET_SIZE:
            flat = flat[:TARGET_SIZE]
        elif flat.shape[0] < TARGET_SIZE:
            flat = np.pad(flat, (0, TARGET_SIZE - flat.shape[0]), mode='constant')

        # Reshape to [1, 128, 128]
        reshaped = flat.reshape(1, TARGET_HEIGHT, TARGET_WIDTH)

        # Save
        npy_path = os.path.join(out_dir, filename.replace(".csv", ".npy"))
        np.save(npy_path, reshaped)
        print(f"Saved {npy_path}, shape: {reshaped.shape}")
