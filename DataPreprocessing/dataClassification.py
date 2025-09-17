import pandas as pd
import numpy as np
import glob
import os
import re
from datetime import datetime, timedelta
import shutil

# Classes:
# 1	Building the contour. The segment of this operation is characterized by rows in the process data sheet with consecutive “contour” indicators.
# 2	Transiting from building contour to building infill. This segment is characterized by the gap between two rows where the indicator transfers from “contour” to “infill” indicators.
# 3	Building the infill. This segment is characterized by rows with consecutive “infill” indicators
# 4	Finalizing the current cylinder. This segment is characterized by the gap between two rows where the indicator transfers from “infill” and “contour” indicators. During this operation, the nozzle either moves from the starting point to the next cylinder or waits for the adjustment of the z-axis of the platform so that a new layer is started.

#### remove gap from time tensor ####################################################################################

# Load the original file
df = pd.read_csv('/Users/xluan3/Desktop/Projects/timetensor.csv')

# Offset to add
offset = 8125.357063

# Flag to start applying offset
start_adding = False

# Iterate over the DataFrame and apply offset from Layer 55, Cylinder 1
for idx, row in df.iterrows():
    if not start_adding and row['Layer'] == 55 and row['Cylinder'] == 1:
        start_adding = True
    if start_adding:
        df.loc[idx, ['Class1', 'Class2', 'Class3', 'Class4']] += offset

# Save the updated CSV
df.to_csv('/Users/xluan3/Desktop/Projects/timetensor_nogap.csv', index=False)

##### calculate the time stamp ######################################################################################
df = pd.read_csv('/Users/xluan3/Desktop/Projects/timetensor_nogap.csv')

# Define initial start and resume times
start_datetime = datetime.strptime('2021-06-11 19:23:26', '%Y-%m-%d %H:%M:%S')
resume_time = datetime.strptime('2021-06-11 22:55:22', '%Y-%m-%d %H:%M:%S')

# Define stop and resume markers
# stop_point = {'Layer': 53, 'Cylinder': 7, 'Class': 'Class4'}
stop_point = {'Layer': 54, 'Cylinder': 9, 'Class': 'Class4'}
resume_point = {'Layer': 55, 'Cylinder': 1, 'Class': 'Class1'}

# Prepare a long-format DataFrame
long_df = df.melt(id_vars=['Layer', 'Cylinder'],
                  value_vars=['Class1', 'Class2', 'Class3', 'Class4'],
                  var_name='Class', value_name='Seconds')

# Sort properly
long_df.sort_values(by=['Layer', 'Cylinder', 'Class'], inplace=True)
long_df.reset_index(drop=True, inplace=True)

# Initialize variables
timestamps = []
paused = False
resumed = False

# Start tracking from start_datetime
base_time = start_datetime

for idx, row in long_df.iterrows():
    # If not yet paused
    if not paused:
        # Calculate timestamp relative to start_datetime
        time_delta = timedelta(seconds=(row['Seconds']))
        current_timestamp = start_datetime + time_delta
        timestamps.append(current_timestamp)

        # Check if we hit the stop point
        if (row['Layer'] == stop_point['Layer'] and
                row['Cylinder'] == stop_point['Cylinder'] and
                row['Class'] == stop_point['Class']):
            paused = True
    else:
        # After pause, look for resume point
        if (row['Layer'] == resume_point['Layer'] and
                row['Cylinder'] == resume_point['Cylinder'] and
                row['Class'] == resume_point['Class']):
            # Reset baseline from resume_time
            base_time = resume_time
            base_seconds_resume = row['Seconds']
            current_timestamp = resume_time
            timestamps.append(current_timestamp)
            resumed = True
        elif resumed:
            # After resuming
            time_delta = timedelta(seconds=(row['Seconds'] - base_seconds_resume))
            current_timestamp = resume_time + time_delta
            timestamps.append(current_timestamp)
        else:
            # During pause but before resume
            timestamps.append(None)

# Assign timestamps to DataFrame
long_df['Timestamp'] = timestamps

# Define the mapping
class_mapping = {
    'Class1': 1,
    'Class2': 2,
    'Class3': 3,
    'Class4': 4
}

# Apply the mapping to the 'Class' column
long_df['Class'] = long_df['Class'].map(class_mapping)

# Save to a new CSV file
long_df.to_csv('/Users/xluan3/Desktop/Projects/timestamp.csv', index=False)

##### AE data classification ######################################################################################

timestamp_path = '/Users/xluan3/Desktop/Projects/timestamp.csv'
timestamp_df = pd.read_csv(timestamp_path)

projpath = '/Users/xluan3/Desktop/Projects/AE_data'
ae_files = [f for f in os.listdir(projpath) if f.endswith('.csv') and 'FA4stream' in f]

start_time = datetime.strptime("2021-06-11 19:23:26", "%Y-%m-%d %H:%M:%S")

# Convert seconds into actual datetime
timestamp_df['Timestamp'] = timestamp_df['Seconds'].apply(lambda x: start_time + timedelta(seconds=x))

# Extract timestamp from AE filenames and convert to datetime
ae_data = []
for file in ae_files:
    time_str = file.split('-')[1].split('.')[0]
    ae_time = datetime.strptime(f"2021-06-11 {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}", "%Y-%m-%d %H:%M:%S")
    ae_data.append({'Filename': file, 'AE_Timestamp': ae_time})

ae_df = pd.DataFrame(ae_data)

# Assign metadata from timestamp_df to AE files
assigned_data = []
for _, ae_row in ae_df.iterrows():
    ae_time = ae_row['AE_Timestamp']
    match = timestamp_df[timestamp_df['Timestamp'] <= ae_time].sort_values('Timestamp', ascending=False).head(1)
    if not match.empty:
        match_row = match.iloc[0]
        assigned_data.append({
            'Filename': ae_row['Filename'],
            'AE_Timestamp': ae_time,
            'Layer': match_row['Layer'],
            'Cylinder': match_row['Cylinder'],
            'Class': match_row['Class']
        })

assigned_df = pd.DataFrame(assigned_data)
print(assigned_df)

# to filter out the " dinner break "
def remove_assignment_during_gap(df, start_str, end_str):
    start_gap = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    end_gap = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
    return df[~df['AE_Timestamp'].between(start_gap, end_gap)]
# Remove assignments between 21:36:38 and 22:55:21 on 2021-06-11
assigned_df = remove_assignment_during_gap(assigned_df, "2021-06-11 21:36:38", "2021-06-11 22:55:21")
print(assigned_df)

# Save to file if needed
assigned_df.to_csv(os.path.join(projpath, 'classified_ae_files.csv'), index=False)

##### AE file storage by classes ######################################################################################
# Paths
base_path = '/Users/xluan3/Desktop/Projects/AE_data'
input_csv = os.path.join(base_path, 'classified_ae_files.csv')
contour_folder = os.path.join(base_path, 'contour_AEdata')
infill_folder = os.path.join(base_path, 'infill_AEdata')

# clear all files in a folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Clear both folders
clear_folder(contour_folder)
clear_folder(infill_folder)

# Make sure destination folders exist
os.makedirs(contour_folder, exist_ok=True)
os.makedirs(infill_folder, exist_ok=True)

# Load classification file
df = pd.read_csv(input_csv)

# Copy files to the folders
for _, row in df.iterrows():
    filename = row['Filename']
    ae_class = row['Class']
    src = os.path.join(base_path, filename)

    if ae_class == 1:
        dst = os.path.join(contour_folder, filename)
        shutil.copy(src, dst)
    elif ae_class == 3:
        dst = os.path.join(infill_folder, filename)
        shutil.copy(src, dst)

####### select layer 71 to 92 ########################################################

# Base project path
base_path = '/Users/xluan3/Desktop/Projects/AE_data'

# Input CSV with classification
input_csv = os.path.join(base_path, 'classified_ae_files.csv')

# Destination folders
contour_dst = os.path.join(base_path, 'contour_AEdata_71to92')
infill_dst = os.path.join(base_path, 'infill_AEdata_71to92')

# Create folders if they don’t exist
os.makedirs(contour_dst, exist_ok=True)
os.makedirs(infill_dst, exist_ok=True)

# Load data
df = pd.read_csv(input_csv)

# Filter for Layer 71–92 and Class 1 or 3
filtered_df = df[(df['Layer'].between(71, 92)) & (df['Class'].isin([1, 3]))]

# Copy files to their respective folders
for _, row in filtered_df.iterrows():
    filename = row['Filename']
    ae_class = row['Class']
    src_path = os.path.join(base_path, filename)

    if ae_class == 1:
        shutil.copy(src_path, os.path.join(contour_dst, filename))
    elif ae_class == 3:
        shutil.copy(src_path, os.path.join(infill_dst, filename))

####### select layer 1 to 20 ########################################################

# Base project path
base_path = '/Users/xluan3/Desktop/Projects/AE_data'

# Input CSV with classification
input_csv = os.path.join(base_path, 'classified_ae_files.csv')

# Destination folders
contour_dst = os.path.join(base_path, 'contour_AEdata_1to20')
infill_dst = os.path.join(base_path, 'infill_AEdata_1to20')

# Create folders if they don’t exist
os.makedirs(contour_dst, exist_ok=True)
os.makedirs(infill_dst, exist_ok=True)

# Load data
df = pd.read_csv(input_csv)

# Filter for Layer 71–92 and Class 1 or 3
filtered_df = df[(df['Layer'].between(1, 21)) & (df['Class'].isin([1, 3]))]

# Copy files to their respective folders
for _, row in filtered_df.iterrows():
    filename = row['Filename']
    ae_class = row['Class']
    src_path = os.path.join(base_path, filename)

    if ae_class == 1:
        shutil.copy(src_path, os.path.join(contour_dst, filename))
    elif ae_class == 3:
        shutil.copy(src_path, os.path.join(infill_dst, filename))