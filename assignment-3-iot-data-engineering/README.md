# Assignment 3: IoT Data Engineering (Batch Processing, Late Events & Anomaly Detection)

## Objective
Build a batch-oriented data pipeline for a stream of IoT sensor readings that:

1. Generates a large synthetic sensor dataset (100,000 records) **in batches**, with anomalies and late-arriving events deliberately injected.
2. Processes that dataset **without ever loading the whole file into memory**, detecting:
   - **Late-arriving / backdated events** — a reading whose timestamp is older than the previous reading from the same device.
   - **Anomalous readings** — values that deviate from the dataset mean by more than 2 standard deviations (Z-Score method).
3. Produces a combined, fully-flagged report plus on-screen summary statistics.

The whole pipeline is written in plain Python + pandas, with batching as the central design constraint.

---

## Architecture Overview

```
generate_data.py
 -> builds 100,000 records in memory-safe batches of 10,000
 -> injects 1% anomalies + 0.5% late events at random positions
 -> writes/appends each batch to CSV
        |
        v
IOT_Data/sensor_data.csv
 (Device_ID, Timestamp, Reading_Value)
        |
        v
main.py  -- PASS 1 (streaming) --
 -> reads batch by batch
 -> accumulates count / sum / sum-of-squares
 -> computes global Mean and Std Dev
        |
        v
main.py  -- PASS 2 (streaming) --
 -> re-reads batch by batch
 -> flags late events (per-device timestamp comparison,
    carried across batch boundaries)
 -> flags anomalies (|Z-Score| > 2)
 -> classifies each row into an Issue_Type
 -> appends each processed batch to the report
        |
        v
IOT_Data/full_report.csv
 (+ Z_Score, Is_Late_Event, Is_Anomaly, Issue_Type)
```

---

## Repository Structure

```
assignment-3-iot-data-engineering/
├── batch_utils.py            # Shared batch config + CSV read/write helpers
├── generate_data.py          # Synthetic IoT data generator (batched)
├── main.py                   # Two-pass batch processor / detector
├── README.md                 # this file
└── IOT_Data/                 # Output folder (files created at runtime)
    ├── sensor_data.csv       # Generated raw sensor data
    └── full_report.csv       # Combined flagged report
```

> `IOT_Data/` ships empty — both CSVs are produced by running the scripts.

---

## Shared Configuration (`batch_utils.py`)

A single module holds the settings used by **both** the writer and the reader, so the batch size can never drift between the two scripts.

| Constant | Value | Purpose |
|---|---|---|
| `BATCH_SIZE` | `10000` | Records per batch (generation and processing) |
| `DATA_DIR` | `IOT_Data` | Output folder |
| `SENSOR_FILE` | `IOT_Data/sensor_data.csv` | Generated raw data |
| `REPORT_FILE` | `IOT_Data/full_report.csv` | Final flagged report |

Two helpers:

- **`write_batch(df, file_path, first_batch)`** — creates the file with a header on the first batch, appends without a header on every batch after that.
- **`read_batches(file_path, batch_size)`** — a generator built on `pd.read_csv(..., chunksize=...)` that yields one batch at a time and parses `Timestamp` into a real datetime before yielding.

---

## Step 1 — Data Generation (`generate_data.py`)

### Dataset shape

| Item | Value |
|---|---|
| Total records | 100,000 |
| Devices | `D001`–`D005` (round-robin, `i % 5`) |
| Start timestamp | `2026-07-31 10:00` |
| Timestamp interval | 1 minute per record (so each device reports every 5 minutes) |
| Normal reading range | `24.0` – `30.0` |
| Columns | `Device_ID`, `Timestamp`, `Reading_Value` |

### Injected issues

Both are expressed as a **percentage of total records**, so the dataset scales automatically — no record positions are hard-coded.

| Issue | Percentage | Count | How it is injected |
|---|---|---|---|
| Anomaly | 1% | 1,000 | Reading pushed outside the normal band — either high (`40.0`–`55.0`) or low (`5.0`–`15.0`), chosen 50/50 |
| Late event | 0.5% | 500 | Timestamp backdated by a random 10–120 minutes, so it lands before that device's previous reading |

Both index sets are drawn from **one** `random.sample()` call over `range(total_records)`. Since a single sample never repeats a value, the two groups are internally unique *and* guaranteed not to overlap. They are stored as **sets** so the membership check inside the record loop stays O(1) — with 100,000 records against 1,500 indices, list lookups would be noticeably slow.

### Batching

Records are built 10,000 at a time, converted to a DataFrame, and written with `write_batch()` (create on batch 1, append afterwards). Any pre-existing `sensor_data.csv` is deleted first so runs don't append onto each other.

---

## Step 2 — Batch Processing & Detection (`main.py`)

The processor makes **two streaming passes** over the CSV. Only one batch is resident in memory at any moment.

### Why two passes?

The Z-Score needs the mean and standard deviation of the **whole** file. Those aren't known until every record has been seen, so no row can be judged during the first read.

### Pass 1 — Collect statistics

Accumulates only three running totals across batches:

- `total_count`
- `total_sum`
- `total_sum_sq` (sum of squares)

Then computes:

```
mean     = total_sum / total_count
variance = (total_sum_sq - total_count * mean^2) / (total_count - 1)
std      = sqrt(variance)
```

Sample variance (`ddof=1`) is used, matching pandas' default `.std()`.

### Pass 2 — Flag every record

**a) Late-arriving / backdated events**

- Within a batch, the previous reading per device comes from `groupby("Device_ID")["Timestamp"].shift(1)`.
- For the *first* row of a device in a batch, `shift(1)` yields `NaT`, so it is filled from a `last_timestamp` dict carrying each device's last seen timestamp from the **previous** batch. This is what keeps detection correct when a device's readings straddle a batch boundary.
- A row is late when `Timestamp < previous Timestamp` for that device.
- After flagging, `last_timestamp` is updated with each device's final timestamp in the batch.

**b) Anomaly detection (Z-Score)**

```
Z_Score = (Reading_Value - mean) / std
```

A row is anomalous when `|Z_Score| > 2` (threshold = 2 standard deviations).

**c) Combined classification**

Each row gets an `Issue_Type`:

| Is_Late_Event | Is_Anomaly | Issue_Type |
|---|---|---|
| ✅ | ✅ | `Late Event + Anomaly` |
| ✅ | ❌ | `Late Event` |
| ❌ | ✅ | `Anomaly` |
| ❌ | ❌ | `Normal` |

**d) Output**

- Every processed batch is appended to `IOT_Data/full_report.csv` (raw columns + `Z_Score`, `Is_Late_Event`, `Is_Anomaly`, `Issue_Type`).
- Only the **flagged** rows are retained in memory and concatenated at the end for the on-screen late-event and anomaly listings — the normal rows are never held.

---

## How to Run

### Requirements

- Python 3
- `pandas`, `numpy`

```bash
pip install pandas numpy
```

### 1. Generate the sensor data

```bash
python generate_data.py
```

Prints a per-batch trace, the exact record index of every injected anomaly / late event, and a final summary:

```
======================================
Sensor Data Generated Successfully!
Total Records: 100000
Batch Size: 10000
Total Batches: 10
Anomalies Injected: 1000 (1%)
Late Events Injected: 500 (0.5%)
CSV Saved: IOT_Data/sensor_data.csv
======================================
```

### 2. Process the data and detect issues

```bash
python main.py
```

Prints, in order:

- the first 5 records of the dataset,
- the full list of late-arriving / backdated events (`Device_ID`, `Timestamp`, `Reading_Value`),
- the computed **Mean** and **Std Dev**,
- the full list of detected anomalies including each `Z_Score`,
- a summary of total records, late events found and anomalies found,
- batch size, total batch count, and the report path.

Both scripts must be run from the assignment folder, since `IOT_Data/` is a relative path.

---

## Output Schema (`IOT_Data/full_report.csv`)

| Column | Type | Description |
|---|---|---|
| `Device_ID` | string | Sensor device (`D001`–`D005`) |
| `Timestamp` | datetime | Reading time (backdated for injected late events) |
| `Reading_Value` | float | Sensor reading |
| `Z_Score` | float | `(value - mean) / std` |
| `Is_Late_Event` | bool | Timestamp older than the device's previous reading |
| `Is_Anomaly` | bool | `|Z_Score| > 2` |
| `Issue_Type` | string | `Normal` / `Late Event` / `Anomaly` / `Late Event + Anomaly` |

---

## Notes / Learnings

- **Batching is the design constraint, not an afterthought.** Both generation and processing move 10,000 rows at a time, so peak memory stays flat regardless of `total_records` — the same code handles 100,000 or 10,000,000 records.
- **Global statistics from a stream.** Mean and standard deviation are derived from running `count` / `sum` / `sum-of-squares` accumulators, which is why the whole file never needs to be resident. This forces the two-pass structure.
- **Batch boundaries are where streaming logic usually breaks.** A naive per-batch `groupby().shift(1)` silently misses any device whose readings split across two batches. The `last_timestamp` carry-over dict is what makes late-event detection correct rather than approximately correct.
- **Injection points are percentage-driven and set-based** — the dataset scales without touching code, indices can't collide between the anomaly and late-event groups, and lookups stay O(1) inside a 100,000-iteration loop.
- **Only flagged rows are kept.** The report is written incrementally per batch; in-memory retention is limited to the ~1.5% of rows that are actually anomalous or late.
- **Threshold choice.** `|Z| > 2` is deliberately sensitive. Because injected anomalies sit far outside the 24–30 band, they dominate the standard deviation, so the reported anomaly count can differ slightly from the 1,000 injected — that gap is the detector's real-world behaviour, not a bug.

---

## Author
Tejas
