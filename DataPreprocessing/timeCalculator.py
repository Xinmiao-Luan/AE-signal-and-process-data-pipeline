from datetime import datetime

# Define the base time
base_time = datetime.strptime("2021-06-11 19:21:06.5", "%Y-%m-%d %H:%M:%S.%f")

# Your list of timestamps
timestamps = [
    "2021-06-11 19:25:36",
    "2021-06-11 19:25:41",
    "2021-06-11 19:25:44",
    "2021-06-11 19:27:12",
    "2021-06-11 19:27:33",
    "2021-06-11 19:26:21"

]

# Convert and calculate differences
time_deltas = []
for ts in timestamps:
    current_time = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    delta_seconds = (current_time - base_time).total_seconds()
    time_deltas.append(delta_seconds)

print(time_deltas)