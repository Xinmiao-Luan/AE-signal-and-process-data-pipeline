import os
import numpy as np
import pandas as pd

folder = "/Users/xluan3/Desktop/Projects/spectrograms"

# Thresholds for silence detection
silence_threshold = 1e-6     # element-wise absolute amplitude threshold

silent_files = []

for fname in os.listdir(folder):
    if fname.endswith(".csv"):
        path = os.path.join(folder, fname)

        try:
            # Each file is expected to be a 200x98 matrix
            df = pd.read_csv(path, header=None)
            arr = df.values

            # if arr.shape != (200, 98):
            #     print(f"Skipping {fname}: unexpected shape {arr.shape}")
            #     continue

            # Calculate ratio of elements above threshold
            ratio = np.mean(np.abs(arr) > silence_threshold)


        except Exception as e:
            print(f"Could not read {fname}: {e}")

# Final result summary
print("\n=== Summary ===")
if silent_files:
    print(f"Total silent files: {len(silent_files)}")
    for f in silent_files:
        print(f"- {f}")
else:
    print("No silent files detected.")
