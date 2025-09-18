import pandas as pd
import scipy.io
from datetime import timedelta

# Load the CSV file
csv_path = '/Users/xluan3/Desktop/Projects/layer1to5/classified_with_process_porosity_labels.csv'
df = pd.read_csv(csv_path)

# Filter rows where selectLabel == 1
selected_df = df[df['selectLabel'] == 1].copy()

# Load the .mat file with the timetable
# Load the .mat timetable with MATLAB datetime
tt_df = pd.read_csv("/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/tt_export.csv")
tt_list = pd.to_datetime(tt_df['Time']).tolist()

# Drop original 'Time' column if it exists
if 'Time' in selected_df.columns:
    selected_df = selected_df.drop(columns=['Time'])

# Each index spans 2 seconds divided into 98 steps
seconds_per_step = 2.0 / 98

# Compute exact timestamp for each row
timestamps = []
for _, row in selected_df.iterrows():
    index = int(row['Index'])        # 1-based
    step_id = int(row['step_id'])    # 1-based
    base_time = tt_list[index - 1]   # Adjust to 0-based index for Python
    offset = (step_id - 1) * seconds_per_step
    exact_time = base_time + timedelta(seconds=offset)
    timestamps.append(exact_time)

# Add new column 'time'
selected_df['time'] = timestamps

# Save the result
output_path = "/Users/xluan3/Desktop/Projects/layer1to5/selected_timestamps.csv"
selected_df.to_csv(output_path, index=False)
print(f"Saved timestamped results to: {output_path}")


# # layer 1 to 20:
# # Load the CSV file
# csv_path = '/Users/xluan3/Desktop/Projects/layer1to20/remaining_class3_VAEresults and process data.csv'
# df = pd.read_csv(csv_path)
#
# # Filter rows where selectLabel == 1
# selected_df = df[df['selectLabel'] == 1].copy()
#
# # Load the .mat file with the timetable
# # Load the .mat timetable with MATLAB datetime
# tt_df = pd.read_csv("/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/tt_export.csv")
# tt_list = pd.to_datetime(tt_df['Time']).tolist()
#
# # Drop original 'Time' column if it exists
# if 'Time' in selected_df.columns:
#     selected_df = selected_df.drop(columns=['Time'])
#
# # Each index spans 2 seconds divided into 98 steps
# seconds_per_step = 2.0 / 98
#
# # Compute exact timestamp for each row
# timestamps = []
# for _, row in selected_df.iterrows():
#     index = int(row['Index'])        # 1-based
#     step_id = int(row['step_id'])    # 1-based
#     base_time = tt_list[index - 1]   # Adjust to 0-based index for Python
#     offset = (step_id - 1) * seconds_per_step
#     exact_time = base_time + timedelta(seconds=offset)
#     timestamps.append(exact_time)
#
# # Add new column 'time'
# selected_df['time'] = timestamps
#
# # Save the result
# output_path = "/Users/xluan3/Desktop/Projects/layer1to20/selected_timestamps.csv"
# selected_df.to_csv(output_path, index=False)
# print(f"Saved timestamped results to: {output_path}")