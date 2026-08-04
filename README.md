# Data Engineering Technical Assessment

Four self-contained assignments covering event-driven cloud pipelines, advanced SQL,
batch processing of large datasets, and end-to-end predictive modelling.

Each assignment lives in its own folder with its own detailed README. This file is the
index — read it to find the right folder, then read that folder's README for the full
write-up.

| # | Assignment | Focus | Stack |
|---|---|---|---|
| 1 | [Event-Driven S3 Processing](assignment-1-event-driven-s3-processing/README.md) | Serverless CSV → JSON pipeline triggered by S3 events | AWS S3, Lambda (Python 3.12), IAM, CloudWatch |
| 2 | [Advanced SQL Transformations](assignment-2-advanced-sql-transformations/README.MD) | CTEs, window functions, moving averages, cross-granularity reconciliation | PostgreSQL (pgAdmin), pure SQL |
| 3 | [IoT Data Engineering](assignment-3-iot-data-engineering/README.md) | Batched generation + two-pass streaming detection of late events and anomalies | Python, pandas, NumPy |
| 4 | [Predictive Modelling](assignment-4-predictive-modelling/README.md) | Full ML pipeline on the Titanic dataset, in seven phases | Python, pandas, matplotlib, scikit-learn |

---

## Repository layout

```
data-engineering-technical-assessment/
│
├── assignment-1-event-driven-s3-processing/
│   ├── README.md                 # architecture, AWS resources, test walkthrough
│   ├── screenshots/              # input bucket, Lambda config, output bucket
│   └── uploads/test.csv          # sample input file
│
├── assignment-2-advanced-sql-transformations/
│   └── README.MD                 # schema, three SQL tasks, results and rationale
│
├── assignment-3-iot-data-engineering/
│   ├── README.md
│   ├── batch_utils.py            # shared batch config + CSV read/write helpers
│   ├── generate_data.py          # synthetic IoT generator (batched)
│   ├── main.py                   # two-pass batch processor / detector
│   └── IOT_Data/                 # outputs, created at runtime
│
└── assignment-4-predictive-modelling/
    ├── README.md
    ├── main.py                   # orchestrator — runs all 7 phases
    ├── preprocessing.py          # phases 1, 2, 4
    ├── eda.py                    # phase 3
    ├── train.py                  # phase 5
    ├── evaluate.py               # phase 6
    ├── predict.py                # phase 7
    ├── utils.py                  # shared helpers
    ├── requirements.txt
    ├── Data/                     # input dataset + cleaned output
    └── plots/                    # generated charts
```

---

## Assignment 1 — Event-Driven S3 Processing

A serverless pipeline that converts uploaded CSVs to JSON with no servers and no polling.

Uploading a `.csv` file to `s3://input-bucket-csv-2026/uploads/` fires a native S3 event
notification, which invokes the `csv-to-json-processor` Lambda function. The function reads
the object with boto3, parses it via `csv.DictReader`, standardises the headers (lowercase,
spaces → underscores), and writes the JSON array to
`s3://output-bucket-json-2026/uploads/` with the same path and a `.json` extension.

**Highlights**

- Prefix (`uploads/`) and suffix (`.csv`) filters on the event notification, so only the
  intended objects trigger the function.
- Least-privilege IAM: `s3:GetObject` on the input bucket, `s3:PutObject` on the output
  bucket, nothing more.
- Input and output are separate buckets — writing back into the trigger prefix would create
  an infinite invocation loop.
- Object keys are URL-decoded (`urllib.parse.unquote_plus`) because S3 event keys arrive
  encoded.

Verified end to end: `uploads/test.csv` (38 B) in → `uploads/test.json` (129 B) out.
Screenshots of both buckets and the Lambda configuration are in `screenshots/`.

**How to review:** open the assignment README and the three screenshots. Deployment steps
and CloudWatch verification are documented there.

---

## Assignment 2 — Advanced SQL Transformations

Complex data manipulation in **pure SQL** — no cursors, no loops, no stored procedures —
against a banking dataset in PostgreSQL.

Two tables, deliberately at different levels of detail:

- `bank_transactions` — the denormalized fact table (255 rows, 15 customers, 5 branches,
  Jan–Jun 2025). Customer and branch names are repeated on every row on purpose.
- `branch_monthly_summary` — one row per branch per month (~25–30 rows), derived from the
  transactions and then given ±3 % random noise to simulate real reconciliation mismatches.

**Three tasks**

1. **CTE + window functions** — `RANK()` for the top transactions within each account type,
   `LAG()` / `LEAD()` for each customer's previous and next transaction amount. The CTE is
   required because a window function cannot be filtered in the same `SELECT` that defines
   it. Returns 15 rows (top 5 × 3 account types).
2. **Moving average + Top-N** — a 7-transaction rolling `AVG(amount) OVER (PARTITION BY
   branch_id ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`, plus
   `ROW_NUMBER()` in a CTE to pick the top 3 per account type (9 rows). Both `RANK()` and
   `ROW_NUMBER()` are used to show the tie-handling difference.
3. **Cross-granularity join** — a CTE rolls the daily transaction ledger up to monthly
   branch totals, joins it to the reported summary on `branch_id` + `summary_month`, and
   reports `discrepancy` and `discrepancy_pct`. Returns 25 rows; discrepancies land between
   −2.6 % and +2.98 %, matching the injected noise.

**How to review:** the assignment README documents the schema, each query's mechanics, and
the business rationale behind it.

---

## Assignment 3 — IoT Data Engineering

A batch pipeline over 100,000 synthetic IoT sensor readings that never loads the full file
into memory.

```
generate_data.py  →  IOT_Data/sensor_data.csv  →  main.py  →  IOT_Data/full_report.csv
```

**Generation** — 100,000 records across devices `D001`–`D005`, one per minute, written in
batches of 10,000. Issues are injected as a *percentage* of total records so the dataset
scales without code changes: 1 % anomalies (readings pushed to 40–55 or 5–15) and 0.5 %
late events (timestamps backdated 10–120 minutes). Both index sets come from a single
`random.sample()` call, so they can never collide, and are stored as sets for O(1) lookup.

**Processing** — two streaming passes, one batch resident at a time:

- *Pass 1* accumulates `count`, `sum` and `sum-of-squares` to derive the global mean and
  sample standard deviation. The Z-Score needs whole-file statistics, which is exactly why
  a second pass is unavoidable.
- *Pass 2* flags each row. Late events use `groupby("Device_ID")["Timestamp"].shift(1)`,
  with a `last_timestamp` carry-over dict filling the `NaT` at each batch boundary — the
  detail that makes detection correct rather than approximate. Anomalies are `|Z| > 2`.
  Each row is classified `Normal` / `Late Event` / `Anomaly` / `Late Event + Anomaly`.

Only the ~1.5 % of flagged rows are retained in memory; the report is appended per batch.

**Run it** (from inside the assignment folder — `IOT_Data/` is a relative path):

```bash
pip install pandas numpy
python generate_data.py
python main.py
```

---

## Assignment 4 — Predictive Modelling (Titanic)

An end-to-end ML pipeline in seven phases, orchestrated by `main.py`, which contains no
analysis logic of its own and reads as a table of contents.

| Phase | Module | What it does |
|---|---|---|
| 1 | `preprocessing.py` | Load the CSV; inspect shape, dtypes, missing values |
| 2 | `preprocessing.py` | Drop junk columns, fill gaps, rename, one-hot encode, save |
| 3 | `eda.py` | Descriptive stats, grouped survival rates, correlations, charts |
| 4 | `preprocessing.py` | Keep genuinely-labelled rows, split X/y, train-test split, scale |
| 5 | `train.py` | Train Logistic Regression; compare candidates by cross-validation |
| 6 | `evaluate.py` | Accuracy, confusion matrix, classification report, overfitting check |
| 7 | `predict.py` | Predict for 6 invented passengers, with a hand-computed sigmoid trace |

**Results** (reproducible — `random_state=42`):

| Metric | Value |
|---|---|
| Rows used for modelling | 891 of 1309 |
| Baseline (always guess "died") | 61.62 % |
| **Test accuracy (unseen data)** | **80.45 %** |
| Improvement over baseline | +18.83 pp |
| Train vs test gap | 0.31 pp → no overfitting |

**Three decisions that keep the score honest**

1. The supplied dataset stacks Kaggle's `train.csv` (891 real labels) on `test.csv`
   (418 rows whose labels are hidden and were filled with `0`). Phase 4 drops those 418
   fabricated rows — this *lowers* the apparent accuracy but makes it real.
2. `StandardScaler` is fitted on training data only and merely `transform`-ed elsewhere,
   including in Phase 7, so no test information leaks into the fit.
3. Model selection uses 5-fold cross-validation on the *training* set. The test set is
   never consulted while choosing a model.

**Run it** (from inside the assignment folder):

```bash
pip install -r requirements.txt
python main.py
```

`plt.show()` pauses at the end of phases 3, 6 and 7 until the chart window is closed —
expected, not a hang. Outputs land in `plots/` (5 PNGs) and `Data/titanic_cleaned.csv`.

---

## Getting started

Each assignment is independent — there is no shared environment or root-level install.

| Assignment | Prerequisites |
|---|---|
| 1 | An AWS account with S3, Lambda and IAM access (`ap-south-1` in this build) |
| 2 | PostgreSQL with pgAdmin (or any SQL client) |
| 3 | Python 3, `pandas`, `numpy` |
| 4 | Python 3.8+ (developed on 3.13), `pip install -r requirements.txt` |

Both Python assignments use paths relative to their own folder, so run their scripts from
inside the assignment directory.

---

## Author

Tejas
