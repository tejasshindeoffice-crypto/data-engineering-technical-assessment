import pandas as pd
import random
from datetime import datetime, timedelta
import os

from batch_utils import BATCH_SIZE, SENSOR_FILE, write_batch

# ===============================
# List of IoT devices
# ===============================
devices = ["D001", "D002", "D003", "D004", "D005"]

# Starting timestamp
start_time = datetime(2026, 7, 31, 10, 0)

# ===============================
# Batch Configuration
# ===============================
total_records = 100000
batch_size = BATCH_SIZE

file_path = SENSOR_FILE

# ===============================
# Anomaly / Late Event Configuration
# ===============================
# Both are a percentage of total_records, so the dataset scales
# automatically -- no record positions are hard-coded anywhere.
anomaly_percentage = 1        # 1%   of total_records
late_event_percentage = 0.5   # 0.5% of total_records

anomaly_count = int(total_records * anomaly_percentage / 100)
late_event_count = int(total_records * late_event_percentage / 100)

if anomaly_count + late_event_count > total_records:
    raise ValueError(
        "anomaly_percentage + late_event_percentage cannot exceed 100%"
    )

# ===============================
# Pick Injection Points
# ===============================
# One random.sample() call draws every index we need at once. Because a
# single sample never repeats a value, the two groups below are unique
# AND guaranteed not to overlap each other.
injected_indices = random.sample(
    range(total_records),
    anomaly_count + late_event_count
)

# Sets, not lists -- lookup inside the record loop must stay O(1),
# otherwise 100000 records x 1500 indices becomes very slow.
anomaly_indices = set(injected_indices[:anomaly_count])
late_event_indices = set(injected_indices[anomaly_count:])

# Remove old CSV if it exists
if os.path.exists(file_path):
    os.remove(file_path)

# ===============================
# Generate Data in Batches
# ===============================
for batch_start in range(0, total_records, batch_size):

    sensor_data = []

    batch_end = min(batch_start + batch_size, total_records)

    print(f"\n========== Batch {(batch_start // batch_size) + 1} ==========")

    for i in range(batch_start, batch_end):

        # Select Device
        device = devices[i % len(devices)]

        # Generate Timestamp
        timestamp = start_time + timedelta(minutes=i)

        # Generate Reading
        reading = round(random.uniform(24.0, 30.0), 1)

        # ===============================
        # Add Anomalies
        # ===============================
        if i in anomaly_indices:

            # Push the reading well outside the normal 24.0 - 30.0 band,
            # randomly either too high or too low
            if random.random() < 0.5:
                reading = round(random.uniform(40.0, 55.0), 1)
            else:
                reading = round(random.uniform(5.0, 15.0), 1)

            print(f"Anomaly added at Record {i}")

        # ===============================
        # Add Late Arriving / Backdated Events
        # ===============================
        if i in late_event_indices:

            # Backdate the record so it arrives out of order. Each device
            # reports every len(devices) minutes, so going back at least
            # 10 minutes always lands before that device's last reading.
            timestamp = timestamp - timedelta(minutes=random.randint(10, 120))

            print(f"Late Event added at Record {i}")

        # Store Record
        sensor_data.append([device, timestamp, reading])

    # Convert Batch into DataFrame
    df = pd.DataFrame(
        sensor_data,
        columns=["Device_ID", "Timestamp", "Reading_Value"]
    )

    # Save First Batch / Append Remaining Batches
    write_batch(df, file_path, first_batch=(batch_start == 0))

    print(f"Records {batch_start} to {batch_end - 1} written successfully.")

print("\n======================================")
print("Sensor Data Generated Successfully!")
print(f"Total Records: {total_records}")
print(f"Batch Size: {batch_size}")

# Ceiling division, so a partial last batch is still counted
print(f"Total Batches: {(total_records + batch_size - 1) // batch_size}")

print(f"Anomalies Injected: {anomaly_count} ({anomaly_percentage:g}%)")
print(f"Late Events Injected: {late_event_count} ({late_event_percentage:g}%)")
print(f"CSV Saved: {file_path}")
print("======================================")