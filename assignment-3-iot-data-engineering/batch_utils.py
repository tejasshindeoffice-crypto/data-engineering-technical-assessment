import pandas as pd

# ===============================
# Shared Batch Configuration
# ===============================
# Used by both generate_data.py (writing) and main.py (reading)
BATCH_SIZE = 10000

DATA_DIR = "IOT_Data"
SENSOR_FILE = DATA_DIR + "/sensor_data.csv"
REPORT_FILE = DATA_DIR + "/full_report.csv"


# ===============================
# Write One Batch to CSV
# ===============================
def write_batch(df, file_path, first_batch):
    """Create the file on the first batch, append on every batch after that."""

    if first_batch:
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)


# ===============================
# Read CSV One Batch at a Time
# ===============================
def read_batches(file_path, batch_size=BATCH_SIZE):
    """Yield the CSV batch by batch instead of loading the whole file in memory."""

    for batch in pd.read_csv(file_path, chunksize=batch_size):
        batch["Timestamp"] = pd.to_datetime(batch["Timestamp"])
        yield batch
