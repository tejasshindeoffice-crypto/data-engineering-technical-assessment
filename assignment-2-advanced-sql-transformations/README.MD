Database   : banking
Tool used  : pgAdmin (PostgreSQL)
Domain     : Banking

--------------------------------------------------------------------
1. OBJECTIVE
--------------------------------------------------------------------
Evaluate complex data manipulation skills using PURE SQL only
(no cursors, no loops, no stored procedures). The three required
capabilities were:

  1. Use CTEs / Window Functions (RANK, LEAD, LAG) to identify
     patterns in a denormalized dataset.
  2. Calculate Moving Averages and flag Top-N items per category.
  3. Perform a complex join reconciling two datasets that are at
     DIFFERENT levels of granularity.


--------------------------------------------------------------------
2. TABLES CREATED (2 total)
--------------------------------------------------------------------

TABLE 1: bank_transactions
---------------------------
Purpose : The main denormalized table (this is the banking
          equivalent of the "Sales" table example in the
          assignment). One row = one transaction. Customer name
          and branch name are repeated/redundant on every row
          instead of being normalized into separate tables -
          this redundancy is intentional, matching the
          assignment's "denormalized dataset" requirement.

Columns :
  transaction_id     SERIAL PRIMARY KEY   - unique row id
  customer_id        INT                  - which customer
  customer_name      VARCHAR(100)         - redundant/denormalized
  branch_id          INT                  - which branch
  branch_name        VARCHAR(100)         - redundant/denormalized
  account_type       VARCHAR(50)          - Savings / Current / Loan
  transaction_type   VARCHAR(20)          - Deposit / Withdrawal / Transfer
  amount             NUMERIC(12,2)        - transaction amount
  transaction_date   DATE                 - date of transaction

Data    : 255 rows generated for 15 customers across 5 branches,
          spread across Jan 2025 - Jun 2025.


TABLE 2: branch_monthly_summary
---------------------------------
Purpose : A second dataset at a DIFFERENT granularity - one row
          per BRANCH per MONTH (instead of one row per
          transaction). Used specifically for Task 3, to prove
          the ability to reconcile data across mismatched levels
          of detail (daily vs monthly).

Columns :
  branch_id       INT             - which branch
  branch_name     VARCHAR(100)    - branch name
  summary_month   DATE            - first day of the month
  reported_total  NUMERIC(12,2)   - the "reported" total for that
                                    branch/month (generated with a
                                    deliberate +/-3% random
                                    variance from the real total,
                                    to simulate real-world
                                    reconciliation mismatches)

Data    : ~25-30 rows (5 branches x 6 months), auto-calculated
          from bank_transactions and then given random noise to
          create realistic discrepancies.


--------------------------------------------------------------------
3. TASK 1 - CTE + WINDOW FUNCTIONS (RANK, LAG, LEAD)
--------------------------------------------------------------------
Goal: Identify specific patterns in the transaction data.

What it does:
  - RANK()  -> ranks every transaction's amount WITHIN its own
              account_type (Savings/Current/Loan), so we can find
              the top 5 biggest transactions per category.
  - LAG()   -> for each customer, shows their PREVIOUS
              transaction amount (in date order).
  - LEAD()  -> for each customer, shows their NEXT transaction
              amount (in date order).

Why a CTE was needed:
  Window functions (RANK/LAG/LEAD) cannot be filtered directly in
  the same SELECT where they are calculated (illegal in SQL - you
  cannot use "WHERE amount_rank_in_type <= 5" in the same query
  that defines amount_rank_in_type). A CTE solves this in two
  steps: Step 1 (inside the CTE) calculates the ranks, Step 2
  (outer query) filters on the calculated ranks.

Business value:
  - RANK() surfaces top/VIP transactions per account type -
    useful for review or fraud monitoring.
  - LAG/LEAD reveal sudden spikes or drops in a customer's
    transaction behavior over time.

Result: 15 rows returned (top 5 transactions x 3 account types).


--------------------------------------------------------------------
4. TASK 2 - MOVING AVERAGE + TOP-N PER CATEGORY
--------------------------------------------------------------------

PART A: Moving Average
  Goal: Calculate a 7-transaction rolling average of amount, per
        branch, ordered by date.
  How : AVG(amount) OVER (PARTITION BY branch_id
                           ORDER BY transaction_date
                           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
  Why : Smooths out day-to-day spikes and shows the underlying
        trend of money flow through each branch over time.
        (Early rows per branch will show a smaller "average" since
        there aren't 6 prior rows yet - expected, not an error.)

PART B: Top-N per category (no stored procedure)
  Goal: Find the Top 3 highest-value transactions per account_type.
  How : ROW_NUMBER() OVER (PARTITION BY account_type
                            ORDER BY amount DESC) inside a CTE,
        then filtered to rn <= 3 in the outer query.
  Note: ROW_NUMBER() always gives strict 1,2,3... even with tied
        amounts, unlike RANK() (used in Task 1) which allows ties
        to share the same rank number. Both are valid window
        functions - using both in this assignment shows
        understanding of the difference.

Result: Part B returned exactly 9 rows (3 account types x top 3
        each).


--------------------------------------------------------------------
5. TASK 3 - COMPLEX JOIN ACROSS DIFFERENT GRANULARITIES
--------------------------------------------------------------------
Goal: Reconcile two datasets that are NOT at the same level of
      detail - bank_transactions (daily/per-transaction) vs
      branch_monthly_summary (monthly/per-branch).

How it works:
  1. A CTE (monthly_actuals) rolls up bank_transactions from
     daily transaction-level data into monthly totals per branch,
     using SUM(amount) and GROUP BY branch + month. This converts
     the fine-grained daily data into the same granularity as the
     summary table.
  2. The rolled-up monthly_actuals is then JOINed to
     branch_monthly_summary on branch_id AND summary_month.
  3. discrepancy = reported_total - calculated_total
     discrepancy_pct = that difference as a percentage

Business value:
  This mirrors a real bank reconciliation process: comparing what
  a branch "reported" for the month against what the underlying
  transaction ledger actually shows, and flagging mismatches for
  investigation.

Result: 25 rows returned, sorted by largest discrepancy first.
        Discrepancies ranged from about -2.6% to +2.98%, matching
        the deliberate +/-3% noise built into the sample data.


--------------------------------------------------------------------
6. SUMMARY (for viva / explanation)
--------------------------------------------------------------------
"I used a banking dataset instead of a literal Sales table, since
the assignment said 'e.g.' Sales as just one example. I built a
denormalized bank_transactions table (redundant customer/branch
info per row, like a flat Sales table) and a branch_monthly_summary
table at a different granularity (monthly instead of per-
transaction). Using CTEs and window functions - RANK(), LAG(),
LEAD(), ROW_NUMBER(), and AVG() OVER() - I identified top
transactions per account type, tracked each customer's transaction
patterns over time, calculated a 7-transaction moving average per
branch, and reconciled daily transaction totals against monthly
reported summaries to detect discrepancies - all using pure SQL,
without any stored procedures, cursors, or loops."


--------------------------------------------------------------------
7. STATUS
--------------------------------------------------------------------
[x] Denormalized table with redundant info
[x] Task 1 - CTE + RANK/LAG/LEAD
[x] Task 2 - Moving Average
[x] Task 2 - Top-N per category
[x] Task 3 - Complex join across different granularities

ASSIGNMENT COMPLETE.
====================================================================