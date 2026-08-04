# Assignment 1: Event-Driven S3 Processing (CSV → JSON with AWS Lambda)

## Objective
Build a serverless, event-driven pipeline using AWS S3 and Lambda that automatically converts uploaded CSV files into JSON format.

Whenever a new CSV file is uploaded to the **input** S3 bucket, a native S3 Event Notification triggers an AWS Lambda function. The function reads the CSV, converts it to JSON (with standardized headers), and saves the result to a separate **output** S3 bucket.

---

## Architecture Overview

```
[User Upload]
     |
     v
S3 Input Bucket (input-bucket-csv-2026)
 -> folder: uploads/
 -> triggers on: *.csv uploads (All object create events)
     |
     v
S3 Event Notification (trigger-csv-upload)
     |
     v
AWS Lambda Function (csv-to-json-processor)
 -> Reads CSV from input bucket
 -> Converts CSV rows to JSON
 -> Standardizes column headers (lowercase, no spaces)
 -> Writes JSON file to output bucket
     |
     v
S3 Output Bucket (output-bucket-json-2026)
 -> folder: uploads/
 -> Result: *.json
```

---

## AWS Resources Used

### 1. S3 Buckets
- `input-bucket-csv-2026` (Region: `ap-south-1` / Asia Pacific - Mumbai)
- `output-bucket-json-2026` (Region: `ap-south-1` / Asia Pacific - Mumbai)

### 2. IAM Role
- **Name:** `lambda-s3-csv-processor-role`
- **Trusted entity:** AWS Lambda
- **Attached policies:**
  - `AWSLambdaBasicExecutionRole` (AWS managed — CloudWatch logging)
  - `s3-read-write-csv-json-policy` (Customer inline policy)
    - `s3:GetObject` on `input-bucket-csv-2026/*`
    - `s3:PutObject` on `output-bucket-json-2026/*`

### 3. Lambda Function
- **Name:** `csv-to-json-processor`
- **Runtime:** Python 3.12
- **Execution role:** `lambda-s3-csv-processor-role`
- **Trigger:** S3 (`input-bucket-csv-2026`)

### 4. S3 Event Notification
- **Name:** `trigger-csv-upload`
- **Configured on:** `input-bucket-csv-2026`
- **Prefix filter:** `uploads/`
- **Suffix filter:** `.csv`
- **Event type:** All object create events
- **Destination:** Lambda function (`csv-to-json-processor`)

---

## Lambda Function Logic (`lambda_function.py`)

1. Reads the S3 event to identify the source bucket and object key.
2. URL-decodes the object key (handles spaces/special characters).
3. Fetches the CSV file content from the input bucket using boto3.
4. Parses the CSV using Python's `csv.DictReader`.
5. Standardizes column headers → lowercase, spaces replaced with underscores (e.g. `Full Name` → `full_name`).
6. Converts the parsed rows into a JSON array.
7. Writes the resulting JSON file to the output bucket, preserving the same folder path and replacing the `.csv` extension with `.json`.
8. Logs progress messages to CloudWatch for traceability.

---

## How to Test

1. Prepare a sample CSV file, e.g. `test.csv`:
   ```csv
   Name,Age,City
   John,25,NYC
   Jane,30,LA
   ```

2. Upload it to:
   ```
   s3://input-bucket-csv-2026/uploads/test.csv
   ```

3. Within a few seconds, the Lambda function is automatically triggered by the S3 event notification.

4. Check the result at:
   ```
   s3://output-bucket-json-2026/uploads/test.json
   ```

5. To verify execution details / debug errors, check:
   **AWS Lambda → csv-to-json-processor → Monitor → View CloudWatch logs**

---

## Screenshots

### 1. Input Bucket — CSV uploaded to `uploads/` folder
![Input Bucket](screenshots/input-buket.png)

### 2. Lambda Function — Configuration & Code
![Lambda Function](screenshots/lambda.png)

### 3. Output Bucket — Transformed JSON generated
![Output Bucket](screenshots/output-buket.png)

---

## Verified Test Result

| Item | Value |
|---|---|
| Input file uploaded | `uploads/test.csv` (38 B) |
| Output file generated | `uploads/test.json` (129 B) |
| Status | ✅ **SUCCESS** — Lambda triggered automatically on upload and correctly transformed CSV to JSON in the output bucket |

---

## Repository Structure

```
assignment-1-event-driven-s3-processing/
├── lambda_function.py        # Lambda source code
├── README.md                 # this file
├── screenshots/
│   ├── input-buket.png       # Input bucket with uploaded test.csv
│   ├── lambda.png            # Lambda function configuration/code
│   └── output-buket.png      # Output bucket with generated test.json
└── uploads/
    └── test.csv               # Sample test input file
```

---

## Notes / Learnings

- S3 event notifications support prefix/suffix filters, which allow triggering Lambda only for specific folders and file types (here, only `.csv` files inside the `uploads/` prefix).
- IAM least-privilege principle was followed: the Lambda execution role only has `GetObject` on the input bucket and `PutObject` on the output bucket, avoiding unnecessary broad permissions.
- Care was taken to avoid writing output back into the same bucket/prefix as the input, which would otherwise risk an infinite trigger loop.
- URL decoding (`urllib.parse.unquote_plus`) is applied to the S3 object key since S3 event keys can be URL-encoded (e.g. spaces become `+`).

---

## Author
Tejas