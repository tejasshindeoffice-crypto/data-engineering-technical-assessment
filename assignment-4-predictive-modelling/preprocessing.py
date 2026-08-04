"""
preprocessing.py
================
Data ingestion and preprocessing for the Titanic project.

This module covers three phases of the pipeline:

    PHASE 1 - Data Ingestion and Profiling      -> load_and_explore_data()
    PHASE 2 - Data Cleaning & Preprocessing     -> clean_data()
    PHASE 4 - Features, Target and Train-Test Split -> prepare_model_data()

Phase 3 (EDA) sits between phases 2 and 4 and lives in eda.py, because
exploring the data is a separate concern from preparing it.
"""

# ---------------------------------------------------------------
# STEP 1: Import the pandas library
# ---------------------------------------------------------------
# 'import pandas as pd' loads the pandas library and gives it the
# short nickname 'pd'. Every pandas function is now called as pd.something()
# This nickname is a universal convention - all Python code everywhere uses 'pd'.
import pandas as pd

# scikit-learn (imported as 'sklearn') is Python's main machine learning
# library. We import only the specific tools we need rather than the
# whole library, which is the normal convention for sklearn.
#   train_test_split -> splits data into a training half and a testing half
#   StandardScaler   -> puts every feature onto the same numeric scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Our own shared helpers for printing formatted section headers.
from utils import print_phase_banner, print_section


# ===============================================================
#
#              PHASE 1: DATA INGESTION AND PROFILING
#
# ===============================================================

def load_and_explore_data(path="Data/train_and_test.csv"):
    """
    PHASE 1 - Load the Titanic CSV and profile it.

    Displays the first 5 rows, the shape, the dataset info, the column
    names and the missing values.

    Returns the raw DataFrame, untouched.
    """

    # ---------------------------------------------------------------
    # STEP 3: Load the CSV file into a DataFrame
    # ---------------------------------------------------------------
    # pd.read_csv() reads a comma-separated file from disk and returns a
    # "DataFrame" - think of it as a spreadsheet/table living inside Python.
    # The path uses a forward slash and is relative to where main.py sits.
    df = pd.read_csv(path)

    # 'df' is the standard variable name for a DataFrame (short for "data frame").

    # ---------------------------------------------------------------
    # STEP 4: Display the FIRST 5 ROWS
    # ---------------------------------------------------------------
    # .head() returns the top rows of the table. Default is 5, but we pass 5
    # explicitly so the code is self-documenting.
    # Purpose: eyeball the data to confirm it loaded into proper columns.
    print_section("1. FIRST 5 ROWS OF THE DATASET", char="=",
                  leading_newline=False)
    print(df.head(5))

    # ---------------------------------------------------------------
    # STEP 5: Display the SHAPE (rows and columns)
    # ---------------------------------------------------------------
    # .shape is NOT a function - it is an attribute, so there are no ().
    # It returns a tuple: (number_of_rows, number_of_columns)
    print_section("2. SHAPE OF THE DATASET", char="=")
    print("Number of rows    :", df.shape[0])   # index 0 = rows
    print("Number of columns :", df.shape[1])   # index 1 = columns

    # ---------------------------------------------------------------
    # STEP 6: Display DATASET INFORMATION
    # ---------------------------------------------------------------
    # .info() prints a technical summary of the table:
    #   - each column's name
    #   - how many NON-NULL (non-missing) values it has
    #   - its data type (int64 = whole number, float64 = decimal, object = text)
    #   - total memory used
    # This is the single most useful command for understanding a new dataset.
    print_section("3. DATASET INFORMATION", char="=")
    df.info()   # note: info() PRINTS by itself, so we do NOT wrap it in print()

    # ---------------------------------------------------------------
    # STEP 7: Display COLUMN NAMES
    # ---------------------------------------------------------------
    # df.columns gives an Index object holding all column names.
    # We wrap it in list() to print it as a clean, readable Python list.
    print_section("4. COLUMN NAMES", char="=")
    print(list(df.columns))

    # ---------------------------------------------------------------
    # STEP 8: Display MISSING VALUES per column
    # ---------------------------------------------------------------
    # This line does two things, read it RIGHT TO LEFT:
    #   df.isnull()  -> builds a same-sized table of True/False
    #                   (True means "this cell is empty/missing")
    #   .sum()       -> adds up each column. In Python True counts as 1,
    #                   False counts as 0, so the sum = number of missing cells.
    missing_values = df.isnull().sum()

    print_section("5. MISSING VALUES IN EACH COLUMN", char="=")

    # We only want to see columns that ACTUALLY have missing data.
    # missing_values > 0 creates a True/False filter, and putting that filter
    # inside [ ] keeps only the rows where the condition is True.
    # This is called "boolean masking" - a core pandas skill.
    missing_only = missing_values[missing_values > 0]

    # .empty is True when the filtered result has zero entries.
    if missing_only.empty:
        print("No missing values found in any column.")
    else:
        print(missing_only)

    # Always show the grand total as a single number for a quick health check.
    print("\nTotal missing cells in entire dataset:", df.isnull().sum().sum())

    return df


# ===============================================================
#
#                 PHASE 2: DATA CLEANING & PREPROCESSING
#
# ===============================================================

def clean_data(df, output_path="Data/titanic_cleaned.csv"):
    """
    PHASE 2 - Clean and preprocess the raw DataFrame.

    Removes useless columns, fills missing values, renames damaged column
    names and one-hot encodes the nominal categorical column.

    Returns the cleaned DataFrame and also saves it to output_path.
    """

    print_phase_banner("PHASE 2: DATA CLEANING", 20, 26)

    # ---------------------------------------------------------------
    # STEP 1: Work on a COPY, never on the original
    # ---------------------------------------------------------------
    # .copy() makes a completely separate duplicate of the DataFrame in memory.
    # WHY: 'df' holds the raw file exactly as it was ingested. If we modify it
    # directly and make a mistake, the only way back is re-reading the CSV.
    # By cleaning a copy, 'df' stays as our untouched reference.
    # NOTE: writing 'data = df' would NOT copy - both names would point to the
    # SAME table, so editing one would silently edit the other.
    data = df.copy()

    print("\nOriginal shape (rows, columns):", data.shape)

    # ---------------------------------------------------------------
    # STEP 2: Remove the useless 'zero' columns
    # ---------------------------------------------------------------
    # Phase 1 revealed 19 columns named 'zero', 'zero.1' ... 'zero.18'.
    # pandas added the .1 .2 suffixes automatically because the CSV header
    # repeated the same name 'zero' 19 times (duplicate names are not allowed).

    # This is a LIST COMPREHENSION - a compact way to build a list with a filter.
    # Read it as: "give me every column name 'col' from data.columns,
    #              but only if that name starts with the text 'zero'."
    zero_columns = [col for col in data.columns if col.startswith("zero")]

    print("\nNumber of 'zero' columns found:", len(zero_columns))

    # PROOF before deletion: never delete a column on a guess - verify it first.
    # data[zero_columns] selects only those columns.
    # .sum() adds each column down -> one total per column.
    # .sum() again adds those totals -> a single grand total.
    # If that grand total is 0, every value in every one of those columns is 0.
    print("Sum of ALL values inside those columns:",
          data[zero_columns].sum().sum())

    # .drop() removes rows or columns and RETURNS A NEW DataFrame.
    #   columns=zero_columns -> the list of column names to remove
    # We must ASSIGN the result back to 'data', otherwise the change is lost,
    # because .drop() does not modify the original table in place.
    data = data.drop(columns=zero_columns)

    # ---------------------------------------------------------------
    # STEP 3: Remove the ID column
    # ---------------------------------------------------------------
    # 'Passengerid' is just a row counter: 1, 2, 3 ... 1309.
    # It carries no information about whether a person survived, and a model
    # could wrongly "memorise" specific ID numbers instead of learning patterns.
    data = data.drop(columns=["Passengerid"])   # note: a LIST, even for one column

    print("Shape after dropping useless columns:", data.shape)
    print("Remaining columns:", list(data.columns))

    # ---------------------------------------------------------------
    # STEP 4: Rename columns to clear, consistent names
    # ---------------------------------------------------------------
    # The raw file has damaged / inconsistent names:
    #   '2urvived' -> clearly meant to be 'Survived' (the S became a 2)
    #   'sibsp'    -> lowercase, while Parch and Pclass are capitalised
    # Renaming now means every later line of code reads naturally and we
    # never mistype '2urvived' and hit a KeyError.
    #
    # .rename() takes a DICTIONARY that maps  old_name : new_name
    data = data.rename(columns={
        "2urvived": "Survived",   # our TARGET - what we want to predict
        "sibsp": "SibSp"          # Siblings + Spouses aboard
    })

    print("\nColumns after renaming:", list(data.columns))

    # ---------------------------------------------------------------
    # STEP 5: What does each surviving column actually mean?
    # ---------------------------------------------------------------
    print_section("MEANING OF EACH COLUMN")
    print("Survived : 0 = died, 1 = survived            <-- TARGET (the answer)")
    print("Pclass   : ticket class  1 = 1st, 2 = 2nd, 3 = 3rd")
    print("Sex      : 0 = male, 1 = female")
    print("Age      : age in years")
    print("SibSp    : number of siblings / spouses aboard")
    print("Parch    : number of parents / children aboard")
    print("Fare     : ticket price paid")
    print("Embarked : port boarded  0 = Cherbourg, 1 = Queenstown, 2 = Southampton")

    # ---------------------------------------------------------------
    # STEP 6: HANDLE MISSING VALUES
    # ---------------------------------------------------------------
    print_section("HANDLING MISSING VALUES")

    # Re-check what is missing now that the junk columns are gone.
    print("Missing values BEFORE cleaning:")
    print(data.isnull().sum())

    # Only 'Embarked' has holes (2 of them).
    #
    # We have three possible strategies:
    #   (a) DROP the rows        -> loses real passengers. Wasteful here.
    #   (b) DROP the column      -> loses a useful feature for 1307 good rows.
    #   (c) FILL (impute) them   -> keeps everything. BEST for only 2 cells.
    #
    # For a CATEGORY (like a port), the sensible fill value is the MODE:
    # the most frequently occurring value. We cannot use the mean/average,
    # because "the average port" is meaningless - ports are labels, not amounts.

    # .mode() returns a Series (there can be ties for most-common),
    # so [0] takes the first / winning value.
    embarked_mode = data["Embarked"].mode()[0]
    print("\nMost common Embarked value (the mode):", embarked_mode)

    # .fillna(value) replaces every NaN in that column with the given value.
    # We assign it back into the column so the change is actually stored.
    data["Embarked"] = data["Embarked"].fillna(embarked_mode)

    # Now that no NaN remains, convert the column from float64 back to int64.
    # WHY it was float in the first place: pandas stores missing values as NaN,
    # and NaN is technically a float, so ONE missing value forces the whole
    # column to become float64. With the holes filled, int64 is correct again.
    data["Embarked"] = data["Embarked"].astype(int)

    print("\nMissing values AFTER cleaning:")
    print(data.isnull().sum())
    print("\nTotal missing cells remaining:", data.isnull().sum().sum())

    # ---------------------------------------------------------------
    # STEP 7: CATEGORICAL ENCODING
    # ---------------------------------------------------------------
    # A machine learning model can only do arithmetic, so every column must be
    # numeric. Our columns already ARE numbers - but "being a number" is not
    # enough. The NUMBERS MUST MEAN SOMETHING when compared with < and >.
    #
    # There are two kinds of categorical data:
    #
    #   ORDINAL  - the categories have a real, meaningful ORDER.
    #              Pclass: 1st class IS better than 2nd, which IS better than 3rd.
    #              Keeping 1 / 2 / 3 is CORRECT - the ordering is genuine.
    #
    #   NOMINAL  - the categories are just labels with NO order.
    #              Embarked: Cherbourg(0), Queenstown(1), Southampton(2).
    #              Southampton is NOT "twice" Queenstown. There is no order at all.
    #              Leaving it as 0/1/2 secretly TEACHES THE MODEL A FALSE FACT:
    #              that the ports sit on a scale where 2 > 1 > 0.
    #
    # The fix for NOMINAL data is ONE-HOT ENCODING: replace one column holding
    # 3 categories with separate yes/no (1/0) columns, one per category.
    print_section("CATEGORICAL ENCODING")

    # --- Case A: Sex - BINARY, already 0/1 -> nothing to do -------------
    # With only two categories, 0 and 1 are simply "off" and "on".
    # There is no false ordering possible, so this is already ideal.
    print("\nSex (binary 0/1) - already correctly encoded:")
    print(data["Sex"].value_counts())

    # --- Case B: Pclass - ORDINAL -> leave as 1/2/3 ---------------------
    print("\nPclass (ordinal 1/2/3) - order is real, so we keep it as-is:")
    print(data["Pclass"].value_counts().sort_index())

    # --- Case C: Embarked - NOMINAL -> ONE-HOT ENCODE -------------------
    print("\nEmbarked (nominal) BEFORE encoding:")
    print(data["Embarked"].value_counts().sort_index())

    # First map the codes to readable letters, so the new columns get clear
    # names like 'Embarked_S' instead of the meaningless 'Embarked_2'.
    # .map() replaces each value using a dictionary lookup.
    data["Embarked"] = data["Embarked"].map({0: "C", 1: "Q", 2: "S"})

    # pd.get_dummies() performs the one-hot encoding:
    #   columns=["Embarked"] -> which column(s) to expand
    #   prefix="Embarked"    -> name the new columns Embarked_C, Embarked_Q, ...
    #   drop_first=True      -> delete the FIRST new column (Embarked_C)
    #   dtype=int            -> produce 1/0 integers instead of True/False
    #
    # WHY drop_first=True? The three columns are perfectly redundant: if a
    # passenger is not Q and not S, they must be C. Keeping all three creates
    # the "dummy variable trap" (perfect multicollinearity), which makes a
    # Logistic Regression's coefficients unstable. Dropping one loses NOTHING:
    #   Q=0, S=0  ->  Cherbourg     (this becomes the baseline)
    #   Q=1, S=0  ->  Queenstown
    #   Q=0, S=1  ->  Southampton
    data = pd.get_dummies(data, columns=["Embarked"],
                          prefix="Embarked", drop_first=True, dtype=int)

    print("\nEmbarked AFTER one-hot encoding - it became these columns:")
    print(data[["Embarked_Q", "Embarked_S"]].head())

    # ---------------------------------------------------------------
    # STEP 8: FINAL VERIFICATION of the cleaned dataset
    # ---------------------------------------------------------------
    print_section("FINAL CLEANED DATASET")

    print("Shape:", data.shape)
    print("\nColumns:", list(data.columns))

    print("\nFirst 5 rows of the cleaned data:")
    print(data.head())

    print("\nData types (all must be numeric for scikit-learn):")
    print(data.dtypes)

    print("\nAny missing values left?", data.isnull().values.any())

    # .duplicated() flags rows that are an exact copy of an earlier row.
    # .sum() counts them. Duplicates can bias a model by over-weighting
    # whatever pattern happens to be repeated.
    print("Duplicate rows found:", data.duplicated().sum())

    # ---------------------------------------------------------------
    # STEP 9: Save the cleaned data to a new file
    # ---------------------------------------------------------------
    # .to_csv() writes the DataFrame back out to disk.
    #   index=False -> do NOT write pandas' row numbers (0,1,2...) as a column,
    #                  otherwise re-loading this file would add junk column.
    # WHY save at all: in real pipelines, cleaning is expensive and is run once.
    # Later stages then read the clean file instead of repeating the work.
    data.to_csv(output_path, index=False)
    print("\nCleaned dataset saved to:", output_path)

    return data


# ===============================================================
#
#         PHASE 4: FEATURES, TARGET AND TRAIN-TEST SPLIT
#
# ===============================================================

def prepare_model_data(data):
    """
    PHASE 4 - Turn the cleaned table into model-ready arrays.

    Selects the genuinely-labelled rows, separates features from the target,
    splits into train/test sets and scales the features.

    Returns a dictionary holding every object the later phases need.
    """

    print_phase_banner("PHASE 4: FEATURES, TARGET & SPLIT", 15, 20)

    # ---------------------------------------------------------------
    # STEP 1: Keep only the rows with GENUINE survival labels
    # ---------------------------------------------------------------
    # During EDA we discovered a serious data-quality problem:
    # this file is the Kaggle 'train.csv' (891 rows, real labels) stacked on
    # top of 'test.csv' (418 rows whose labels are DELIBERATELY HIDDEN).
    # Whoever built the file filled those 418 unknown labels with 0, so the
    # data now claims all 418 of those passengers died. That is not true -
    # their outcome is simply unknown.
    #
    # Training on fabricated labels teaches the model a false pattern, so we
    # keep only the first 891 rows, where every label is verified.
    print_section("1. SELECTING ROWS WITH GENUINE LABELS")

    print("Rows in the full cleaned dataset :", len(data))
    print("Survivors in the first 891 rows  :", data["Survived"][:891].sum())
    print("Survivors in the last  418 rows  :", data["Survived"][891:].sum(),
          " <-- impossible, these labels are fake")

    # .iloc[start:stop] selects rows BY POSITION. Position 0 up to (not
    # including) position 891, i.e. the first 891 rows.
    # .copy() again, so this is an independent table.
    model_data = data.iloc[:891].copy()

    print("\nRows kept for modelling:", len(model_data))

    # Recompute the baseline on this honest subset. This is the score to beat.
    baseline = (model_data["Survived"] == 0).mean() * 100
    print("Survival rate here : {:.2f}%".format(
        model_data["Survived"].mean() * 100))
    print("NEW BASELINE (always guessing 'died'): {:.2f}%".format(baseline))

    # ---------------------------------------------------------------
    # STEP 2: Split the COLUMNS into features (X) and target (y)
    # ---------------------------------------------------------------
    # Naming convention used everywhere in machine learning:
    #   X = the INPUTS  (capital X, because it is a 2-D table of many columns)
    #   y = the OUTPUT  (lowercase y, because it is a 1-D single column)
    print_section("2. SEPARATING FEATURES (X) AND TARGET (y)")

    # X = every column EXCEPT the answer.
    # We must drop 'Survived' - leaving it in would let the model read the
    # answer straight off the input and score a meaningless 100%.
    X = model_data.drop(columns=["Survived"])

    # y = only the answer column.
    y = model_data["Survived"]

    print("X (features) shape :", X.shape, " -> 891 passengers, 8 clues each")
    print("y (target)   shape :", y.shape, " -> 891 answers")

    print("\nFeature columns being used:")
    # enumerate(list, 1) numbers the items starting from 1 instead of 0.
    for i, col in enumerate(X.columns, 1):
        print("  {}. {}".format(i, col))

    print("\nFirst 5 rows of X:")
    print(X.head())

    print("\nFirst 5 values of y:")
    # .tolist() converts the pandas Series into a plain Python list for printing.
    print(y.head().tolist())

    # ---------------------------------------------------------------
    # STEP 3: Split the ROWS into a training set and a testing set
    # ---------------------------------------------------------------
    # train_test_split() shuffles the rows and cuts them into two groups.
    # It returns FOUR objects, always in this exact order:
    #   X_train, X_test, y_train, y_test
    print_section("3. SPLITTING INTO TRAINING AND TESTING SETS")

    X_train, X_test, y_train, y_test = train_test_split(
        X,                    # the features
        y,                    # the matching answers
        test_size=0.2,        # hold back 20% for testing, train on the other 80%
        random_state=42,      # fixes the shuffle so the split is REPRODUCIBLE
        stratify=y            # keep the same survived/died ratio in both halves
    )

    # WHY test_size=0.2?
    #   Too small a test set -> the score is noisy and unreliable.
    #   Too large a test set -> too little data left to learn from.
    #   20% is the standard compromise for a dataset of this size.
    #
    # WHY random_state=42?
    #   The split is random. Without a fixed seed you would get a different
    #   split - and a slightly different accuracy - on every single run,
    #   making results impossible to reproduce or compare. Any number works;
    #   42 is simply a long-standing convention.
    #
    # WHY stratify=y?
    #   Only ~38% of passengers survived. A purely random split could, by bad
    #   luck, put far more survivors in one half than the other. stratify=y
    #   forces BOTH halves to keep the original survived/died proportion, so
    #   the test set is a fair miniature of the whole dataset.

    print("Training set : {} passengers  ({:.0f}% of the data)".format(
        len(X_train), len(X_train) / len(X) * 100))
    print("Testing set  : {} passengers  ({:.0f}% of the data)".format(
        len(X_test), len(X_test) / len(X) * 100))

    print("\nShapes of the four resulting objects:")
    print("  X_train:", X_train.shape, "   y_train:", y_train.shape)
    print("  X_test :", X_test.shape, "   y_test :", y_test.shape)

    # Proof that stratify worked: the survival rate should be nearly identical
    # in the original data, the training set, and the test set.
    print("\nSurvival rate in each set (proof that stratify=y worked):")
    print("  Original : {:.2f}%".format(y.mean() * 100))
    print("  Training : {:.2f}%".format(y_train.mean() * 100))
    print("  Testing  : {:.2f}%".format(y_test.mean() * 100))

    # ---------------------------------------------------------------
    # STEP 4: FEATURE SCALING
    # ---------------------------------------------------------------
    # Our features live on wildly different scales:
    #   Sex   ranges 0 to 1
    #   Fare  ranges 0 to 512
    # Logistic Regression finds its weights by numerical optimisation, and
    # a feature with huge numbers can distort that process purely because of
    # its UNITS, not because it is genuinely more important.
    #
    # StandardScaler rescales each column to mean = 0 and std = 1 using
    #     z = (value - mean) / standard_deviation
    # The SHAPE of each distribution is unchanged - only the units move.
    print_section("4. FEATURE SCALING")

    print("Feature ranges BEFORE scaling:")
    # .agg([...]) applies several functions at once to every column.
    print(X_train.agg(["min", "max", "mean"]).T.round(2))

    # Create the scaler object.
    scaler = StandardScaler()

    # *** THE MOST IMPORTANT LINE IN THIS PHASE ***
    # .fit_transform() does TWO things on the TRAINING data:
    #   fit()       -> learn each column's mean and standard deviation
    #   transform() -> apply the formula using what it just learned
    X_train_scaled = scaler.fit_transform(X_train)

    # For the test data we call transform() ONLY - never fit().
    # WHY: fitting on the test set would let information from the test data
    # leak into our preparation, and the test set must stay completely unseen.
    # We reuse the mean/std learned from TRAINING, exactly as we would have to
    # for a brand-new passenger in the real world. This is called avoiding
    # DATA LEAKAGE, and getting it wrong silently inflates your final score.
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature ranges AFTER scaling (training set):")
    # fit_transform returns a plain NumPy array (it loses the column names),
    # so we rebuild a DataFrame just to print it in a readable form.
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    print(X_train_scaled_df.agg(["min", "max", "mean"]).T.round(2))

    print("\nEvery column now has mean ~0 and a comparable range.")
    print("\nData is ready for model training.")

    # Bundle everything the later phases need into one dictionary, so main.py
    # can pass the pieces along without juggling eleven separate variables.
    return {
        "model_data": model_data,
        "baseline": baseline,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }
