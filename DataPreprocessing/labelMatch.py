import pandas as pd
import numpy as np

# Load data
timestamp_df = pd.read_csv("/Users/xluan3/Desktop/Projects/timestamp.csv")
porosity_df = pd.read_csv("/Users/xluan3/Desktop/Projects/Process_data_and_Porosity_data.csv")

# Clean time columns
timestamp_df['Seconds'] = timestamp_df['Seconds'].astype(float).round(6)
porosity_df['Time [s]'] = porosity_df['Time [s]'].astype(float).round(6)

# Convert timestamp to datetime
timestamp_df['Timestamp'] = pd.to_datetime(timestamp_df['Timestamp'])
restart_time = pd.to_datetime("2021-06-11 22:55:22")

# Split timestamp into before/after restart
before_restart = timestamp_df[timestamp_df['Timestamp'] < restart_time].copy()
after_restart = timestamp_df[timestamp_df['Timestamp'] >= restart_time].copy()

# Split porosity data into two parts as well
porosity_df_before = porosity_df.iloc[:26951].copy()
porosity_df_after = porosity_df.iloc[26951:].copy()

# Adjust Seconds after restart
restart_seconds = after_restart['Seconds'].iloc[0]
after_restart['Adjusted Seconds'] = after_restart['Seconds'] - restart_seconds

# Function to find nearest porosity record for each timestamp
def assign_nearest_porosity(target_df, reference_df, target_time_col, ref_time_col):
    target_times = target_df[target_time_col].values
    ref_times = reference_df[ref_time_col].values

    nearest_indices = np.abs(ref_times[:, None] - target_times).argmin(axis=1)
    matched_porosity = reference_df.iloc[np.arange(len(reference_df)), :].copy()
    matched_porosity['Nearest_Time'] = target_times[nearest_indices]
    matched_porosity['Porosity_Assigned'] = reference_df['if it is porosity'].values

    # Merge porosity values into target_df based on nearest match
    merged_df = target_df.copy()
    merged_df['Porosity_Assigned'] = np.nan

    for i, t in enumerate(target_times):
        # find closest porosity time index
        idx = np.abs(ref_times - t).argmin()
        merged_df.loc[merged_df[target_time_col] == t, 'Porosity_Assigned'] = reference_df.iloc[idx]['if it is porosity']

    return merged_df

# Match porosity to timestamp records (nearest match)
before_matched = assign_nearest_porosity(before_restart, porosity_df_before, 'Seconds', 'Time [s]')
after_matched = assign_nearest_porosity(after_restart, porosity_df_after, 'Adjusted Seconds', 'Time [s]')

# Drop 'Adjusted Seconds' column for consistency
after_matched.drop(columns=['Adjusted Seconds'], inplace=True)

# Combine and sort
final_df = pd.concat([before_matched, after_matched])
final_df_sorted = final_df.sort_values(by='Timestamp').reset_index(drop=True)

# Save
final_df_sorted.to_csv("/Users/xluan3/Desktop/Projects/timestamp_with_porosity_updated.csv", index=False)
