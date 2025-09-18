import scipy.io
import pandas as pd
from datetime import datetime, timedelta
import os

# Load the .mat file
mat_data = scipy.io.loadmat('/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/proctimesummary.mat')  # <-- Update path
record = mat_data['record']  # shape: (3312, 6)
timetensor = mat_data['timetensor']

mask = (record[:, 0] >= 1) & (record[:, 0] <= 20)
filtered_record = record[mask]

# Extract relative seconds from column 6 (index 5)
relative_seconds = filtered_record[:, 5]

# Base time for alignment
base_time = datetime.strptime('2021-06-11 19:21:06.5', '%Y-%m-%d %H:%M:%S.%f')

# Create aligned datetime list
time_align = [base_time + timedelta(seconds=float(sec)) for sec in relative_seconds]

# Convert to pandas datetime index
time_align = pd.to_datetime(time_align)

# Optional: save to CSV
# time_align.to_series().to_csv("/Users/xluan3/Desktop/Projects/time_align_layer1to20.csv", index=False)

# Print preview
print(time_align[:5])