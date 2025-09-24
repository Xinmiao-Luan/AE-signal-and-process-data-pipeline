import os
import numpy as np
import pandas as pd


folder = "/Users/xluan3/Desktop/Projects/spectrograms"

# Threshold for "silence
silence_threshold = 1e-6   # adjust depending on your data scale
min_nonzero_ratio = 0.01   # at least 1% of values should be > threshold

silent_files = []

for fname in os.listdir(folder):
    if fname.endswith(".csv"):
        path = os.path.join(folder, fname)

        try:
            df = pd.read_csv(path)
            arr = df.values.flatten()

            # ratio of values above threshold
            ratio = np.mean(np.abs(arr) > silence_threshold)

            if ratio < min_nonzero_ratio:
                silent_files.append(fname)
                print(f"Silent file detected: {fname} (non-zero ratio={ratio:.4f})")

        except Exception as e:
            print(f"Could not read {fname}: {e}")

# result
if silent_files:
    print(f"Total silent files: {len(silent_files)}")
    for f in silent_files:
        print(f"- {f}")
else:
    print("No silent files detected.")
