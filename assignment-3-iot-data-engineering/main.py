import pandas as pd
import numpy as np

from batch_utils import BATCH_SIZE, SENSOR_FILE, REPORT_FILE, read_batches, write_batch

# ===============================
# Pass 1: Read in Batches and Collect Statistics
# ===============================
# The Z-Score needs the mean and std of the WHOLE file, so we cannot flag
# anomalies until every record has been seen. Pass 1 only accumulates the
# running totals needed for mean / std -- one batch is in memory at a time.

total_count = 0
total_sum = 0.0
total_sum_sq = 0.0

first_batch = True

for batch in read_batches(SENSOR_FILE, BATCH_SIZE):

    if first_batch:
        print("===== First 5 Records =====")
        print(batch.head())
        first_batch = False

    readings = batch["Reading_Value"]

    total_count += len(readings)
    total_sum += readings.sum()
    total_sum_sq += (readings ** 2).sum()

mean_val = total_sum / total_count

# Sample variance (ddof=1), same as pandas .std()
variance = (total_sum_sq - total_count * mean_val ** 2) / (total_count - 1)
std_val = np.sqrt(variance)

# ===============================
# Pass 2: Read in Batches Again and Flag Each Record
# ===============================
threshold = 2

# Last timestamp seen per device, carried across batch boundaries so that
# late-event detection still works for a device split over two batches.
last_timestamp = {}

late_event_batches = []
anomaly_batches = []

total_records = 0
late_count = 0
anomaly_count = 0

first_batch = True

for batch in read_batches(SENSOR_FILE, BATCH_SIZE):

    # ===============================
    # Detect Late-Arriving / Backdated Events
    # ===============================
    # Previous reading of the device inside this batch...
    prev_timestamp = batch.groupby("Device_ID")["Timestamp"].shift(1)

    # ...and for the first row of a device in this batch, the value carried
    # over from the previous batch.
    carried = pd.to_datetime(batch["Device_ID"].map(last_timestamp))
    prev_timestamp = prev_timestamp.fillna(carried)

    is_late = batch["Timestamp"] < prev_timestamp

    # Remember where each device ended, for the next batch
    last_timestamp.update(
        batch.groupby("Device_ID")["Timestamp"].last().to_dict()
    )

    # ===============================
    # Anomaly Detection (Z-Score)
    # ===============================
    batch["Z_Score"] = (batch["Reading_Value"] - mean_val) / std_val

    is_anomaly = batch["Z_Score"].abs() > threshold

    # ===============================
    # Combine Flags
    # ===============================
    batch["Is_Late_Event"] = is_late
    batch["Is_Anomaly"] = is_anomaly

    def get_issue_type(row):
        if row["Is_Late_Event"] and row["Is_Anomaly"]:
            return "Late Event + Anomaly"
        elif row["Is_Late_Event"]:
            return "Late Event"
        elif row["Is_Anomaly"]:
            return "Anomaly"
        else:
            return "Normal"

    batch["Issue_Type"] = batch.apply(get_issue_type, axis=1)

    # Only the flagged rows are kept in memory for the final report
    late_event_batches.append(batch[is_late])
    anomaly_batches.append(batch[is_anomaly])

    total_records += len(batch)
    late_count += int(is_late.sum())
    anomaly_count += int(is_anomaly.sum())

    # ===============================
    # Write This Batch to the Report
    # ===============================
    write_batch(batch, REPORT_FILE, first_batch)
    first_batch = False

late_events = pd.concat(late_event_batches)
anomalies = pd.concat(anomaly_batches)

# ===============================
# Reports
# ===============================
print("\n===== Late-Arriving / Backdated Events =====")
print(late_events[["Device_ID", "Timestamp", "Reading_Value"]])

print(f"\n===== Stats =====")
print(f"Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")

print("\n===== Detected Anomalies (Z-Score) =====")
print(anomalies[["Device_ID", "Timestamp", "Reading_Value", "Z_Score"]])

print(f"\n===== Summary =====")
print(f"Total records: {total_records}")
print(f"Late-arriving events found: {late_count}")
print(f"Anomalies found: {anomaly_count}")

print(f"\nBatch Size : {BATCH_SIZE}")
print("Total Batches :", (total_records + BATCH_SIZE - 1) // BATCH_SIZE)
print("\nCombined report saved to:", REPORT_FILE)
