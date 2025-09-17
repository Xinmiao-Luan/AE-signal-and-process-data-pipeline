import numpy as np

# File path
csv_path = "/Users/xluan3/Desktop/Projects/layer1to5/result/ae_raw_data_VAE_input.csv"

# Create frequency row
frequencies_kHz = (np.arange(1, 201) - 0.5) * 1.25

# Load the existing data
df = pd.read_csv(csv_path, header=None)

# Prepend frequency row
df_with_freq = pd.concat([pd.DataFrame([frequencies_kHz]), df], ignore_index=True)

# Save back to the same CSV
df_with_freq.to_csv(csv_path, index=False, header=False)
