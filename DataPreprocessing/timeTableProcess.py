import pandas as pd
from datetime import datetime
import os
import shutil

# this is to attach the spectrogram files with labels
# check the time table data
csv_path = "/Users/xluan3/Desktop/Projects/time_stamps for spectrograms.csv"

df = pd.read_csv(csv_path)

print(df.head())


# Load timetable
timetable_path = "/Users/xluan3/Desktop/Projects/time_stamps for spectrograms.csv"
timetable_df = pd.read_csv(timetable_path, parse_dates=["Time"])

# Load timestamp classification table
timestamp_path = "/Users/xluan3/Desktop/Projects/timestamp.csv"
timestamp_df = pd.read_csv(timestamp_path, parse_dates=["Timestamp"])

# Filter timestamp rows for Layer 71–90 only
timestamp_filtered = timestamp_df[(timestamp_df["Layer"] >= 1) & (timestamp_df["Layer"] <= 20)].copy()

# Sort timestamp entries to ensure order
timestamp_filtered.sort_values(by="Timestamp", inplace=True)
timestamp_filtered.reset_index(drop=True, inplace=True)

# Now classify each entry in timetable_df according to class time boundaries
def classify_time(t):
    for i in range(len(timestamp_filtered) - 1):
        t0 = timestamp_filtered.loc[i, "Timestamp"]
        t1 = timestamp_filtered.loc[i + 1, "Timestamp"]
        c0 = timestamp_filtered.loc[i, "Class"]
        c1 = timestamp_filtered.loc[i + 1, "Class"]

        # Class 1: between class 1 and 2
        if c0 == 1 and c1 == 2 and t0 <= t < t1:
            return 1
        # Class 2: between class 2 and 3
        elif c0 == 2 and c1 == 3 and t0 <= t < t1:
            return 2
        # Class 3: between class 3 and 4
        elif c0 == 3 and c1 == 4 and t0 <= t < t1:
            return 3
        # Class 4: between class 4 and next class 1
        elif c0 == 4 and c1 == 1 and t0 <= t < t1:
            return 4
    return None  # if not found

# Apply classification only within the range
start_time = timestamp_filtered["Timestamp"].min()
end_time = timestamp_filtered["Timestamp"].max()

# Filter timetable within time range
in_range_df = timetable_df[(timetable_df["Time"] >= start_time) & (timetable_df["Time"] <= end_time)].copy()
in_range_df['Index'] = in_range_df.index + 1  # 1-based index
# Classify each time
in_range_df["Class"] = in_range_df["Time"].apply(classify_time)

# Drop any rows that couldn't be classified
classified_df = in_range_df.dropna(subset=["Class"])

# Show and save result
print(classified_df[['Index', 'Time', 'Class']].head())
classified_df.to_csv("classified_timetable_1_20.csv", index=False)

## according to the classification, store the spectrum sepeartely
# Paths
base_dir = "/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/Spectro"
csv_path = os.path.join("/Users/xluan3/Desktop/Projects/classified_timetable_1_20.csv")
source_dir = base_dir  # where the .mat files are
class_1_dir = os.path.join("/Users/xluan3/Desktop/Projects/layer1to20", "Class_1")
class_3_dir = os.path.join("/Users/xluan3/Desktop/Projects/layer1to20", "Class_3")

# Create class folders if they don't exist
os.makedirs(class_1_dir, exist_ok=True)
os.makedirs(class_3_dir, exist_ok=True)

# Read CSV
df = pd.read_csv(csv_path)

# Process each row
for _, row in df.iterrows():
    index = str(row['Index'])
    label = row['Class']
    filename = f"{index}.mat"
    src_file = os.path.join(source_dir, filename)

    if not os.path.isfile(src_file):
        print(f"Warning: File {filename} not found. Skipping.")
        continue

    if label == 1:
        shutil.copy(src_file, os.path.join(class_1_dir, filename))
    elif label == 3:
        shutil.copy(src_file, os.path.join(class_3_dir, filename))

# see how many of the spectro?
# Count .mat files in each class folder
num_class_1 = len([f for f in os.listdir(class_1_dir) if f.endswith('.mat')])
num_class_3 = len([f for f in os.listdir(class_3_dir) if f.endswith('.mat')])

print(f"Number of files in Class 1 folder: {num_class_1}")
print(f"Number of files in Class 3 folder: {num_class_3}")