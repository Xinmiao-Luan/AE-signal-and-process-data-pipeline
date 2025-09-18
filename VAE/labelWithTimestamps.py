import pandas as pd
import numpy as np

# expend to 98 time stamps for each spectrogram
# Load original file
input_path = "/Users/xluan3/Desktop/Projects/layer1to20/classified_timetable_1_20.csv"
df = pd.read_csv(input_path)

# Convert 'Time' to datetime
df["Time"] = pd.to_datetime(df["Time"])

# Parameters
n_steps = 98
duration = 2.0  # seconds
step_seconds = np.linspace(0, duration, n_steps)  # 98 time steps in seconds

# Repeat rows
df_expanded = df.loc[df.index.repeat(n_steps)].reset_index(drop=True)

# Repeat original timestamps
repeated_times = pd.to_datetime(np.repeat(df["Time"].values, n_steps))

# Add timestamp steps
time_deltas = pd.to_timedelta(np.tile(step_seconds, len(df)), unit='s')
df_expanded["timestamp"] = repeated_times + time_deltas

# Add step_id
df_expanded["step_id"] = np.tile(np.arange(1, n_steps + 1), len(df))

# Optional: format timestamp nicely
df_expanded["timestamp"] = df_expanded["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

# Save
output_path = "/Users/xluan3/Desktop/Projects/layer1to20/classified_timetable_1_20_expanded.csv"
df_expanded.to_csv(output_path, index=False)

## add the porosity label
# === Load Files ===
# expanded_df = pd.read_csv("/Users/xluan3/Desktop/Projects/layer1to20/classified_timetable_1_20_expanded.csv")
porosity_df = pd.read_csv("/Users/xluan3/Desktop/Projects/Process_data_and_Porosity_data.csv")
expanded_df = df_expanded
# ---- STEP 1: Convert timestamp to seconds ----
def time_to_seconds_relative(t, start_time):
    try:
        t = pd.to_datetime(t)
        delta = (t - start_time).total_seconds()
        return delta
    except:
        return np.nan

expanded_df["timestamp"] = pd.to_datetime(expanded_df["timestamp"])
start_time = expanded_df["timestamp"].iloc[0]

expanded_df["timestamp_sec"] = expanded_df["timestamp"].astype(str).apply(lambda x: time_to_seconds_relative(x, start_time))

# ---- STEP 2: Reference time alignment ----
ref_time_poro = porosity_df.loc[30, "Time [s]"]
ref_time_expanded = expanded_df["timestamp_sec"].iloc[0]

expanded_df["mapped_seconds"] = expanded_df["timestamp_sec"] - ref_time_expanded + ref_time_poro

# ---- STEP 3: Match each expanded row to nearest porosity_df row ----
# Precompute all process-related columns to extract
columns_to_merge = [
    "Time [s]", "if it is porosity", "X [mm]", "Y [mm]", "Z [mm]", "Power [W]",
    "cylinder number", "ContourFlag [0/1]", "InfillFlag [0/1]", "Shield Gas [l/min]",
    "Feeder 1 Carrier Gas [l/min]", "Melt Pool Size [mm]", "Melt Pool Temperature [C]",
    "rho", "phi", "dist_min", "theta"
]

# Function to find nearest full row from porosity_df
def find_nearest_row(t):
    idx = np.abs(porosity_df["Time [s]"].values - t).argmin()
    return porosity_df.loc[idx, columns_to_merge]

# Apply row-wise
matched_data = expanded_df["mapped_seconds"].apply(find_nearest_row)
# matched_df = pd.DataFrame(list(matched_data))  # convert series of rows to DataFrame

# Concatenate with expanded_df
expanded_with_all = pd.concat([expanded_df.reset_index(drop=True), matched_data.reset_index(drop=True)], axis=1)

# ---- STEP 4: Save labeled ----
output_path = "/Users/xluan3/Desktop/Projects/layer1to20/classified_with_process_porosity_labels.csv"
expanded_with_all.to_csv(output_path, index=False)
print(f"Labeled and enriched file saved to: {output_path}")

# ---- STEP 5: Filter Class 3 ----
class3_df = expanded_with_all[expanded_with_all["Class"] == 3].reset_index(drop=True)
# class3_path = "/Users/xluan3/Desktop/Projects/layer1to20/result/layer1to20_class3_label.csv"
# class3_df.to_csv(output_path, index=False)

# ---- STEP 6: Merge with latent_z ----
latent_file = "/Users/xluan3/Desktop/Projects/layer1to20/result/latent_z.csv"
z_2d_projection_file = "/Users/xluan3/Desktop/Projects/layer1to20/result/z_2d_projection.csv"
latent_df = pd.read_csv(latent_file)
z_2d_projection = pd.read_csv(z_2d_projection_file)

# assert class3_df.shape[0] == latent_df.shape[0], "Mismatch between Class 3 rows and latent rows"

merged_df = pd.concat([class3_df.reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)
merged_df = pd.concat([merged_df.reset_index(drop=True), z_2d_projection.reset_index(drop=True)], axis=1)
# Save final merged output
final_output = "/Users/xluan3/Desktop/Projects/layer1to20/result/layer1to20_class3_label.csv"
merged_df = merged_df[merged_df["InfillFlag [0/1]"] == 1].reset_index(drop=True)
merged_df.to_csv(final_output, index=False)
print(f"Final Class 3 + latent file saved to: {final_output}")


