"""
Predictive Modelling with Pandas - Titanic Dataset
====================================================
main.py - the ORCHESTRATOR.

This file does not contain any analysis logic of its own. Its only job is
to call each phase of the pipeline in the correct order and to pass the
results of one phase into the next.

The seven phases live in dedicated modules:

    PHASE 1  Data ingestion & profiling ....... preprocessing.load_and_explore_data()
    PHASE 2  Cleaning & preprocessing ......... preprocessing.clean_data()
    PHASE 3  Exploratory data analysis ........ eda.run_eda()
    PHASE 4  Features, target & split ......... preprocessing.prepare_model_data()
    PHASE 5  Model training ................... train.train_model()
    PHASE 6  Model evaluation ................. evaluate.evaluate_model()
    PHASE 7  Predicting new passengers ........ predict.predict_samples()

Run the whole project with:

    python main.py
"""

# ---------------------------------------------------------------
# Import our own modules
# ---------------------------------------------------------------
# Each module owns one part of the pipeline. Importing them here is what
# lets main.py stay a short, readable table of contents for the project.
import preprocessing
import eda
import train
import evaluate
import predict

# Shared helper that widens pandas' printing so no columns are hidden.
from utils import configure_pandas_display


def main():
    """Run all seven phases of the project, in order."""

    # ---------------------------------------------------------------
    # SETUP: make pandas print ALL columns instead of hiding some
    # ---------------------------------------------------------------
    # This must happen before anything is printed, so every table in every
    # phase is displayed in full.
    configure_pandas_display()

    # ---------------------------------------------------------------
    # PHASE 1: Load the CSV and profile it
    # ---------------------------------------------------------------
    # Returns the RAW DataFrame, exactly as it came off disk.
    df = preprocessing.load_and_explore_data("Data/train_and_test.csv")

    # ---------------------------------------------------------------
    # PHASE 2: Clean and preprocess
    # ---------------------------------------------------------------
    # Takes the raw table, returns the cleaned one (and saves it to CSV).
    data = preprocessing.clean_data(df)

    # ---------------------------------------------------------------
    # PHASE 3: Exploratory Data Analysis
    # ---------------------------------------------------------------
    # Explores the cleaned data and writes two chart images. It does not
    # modify 'data', so nothing is returned that later phases depend on.
    eda.run_eda(data)

    # ---------------------------------------------------------------
    # PHASE 4: Features, target and the train-test split
    # ---------------------------------------------------------------
    # Returns a dictionary holding X, y, the four split objects, the scaled
    # arrays, the fitted scaler and the baseline accuracy.
    prepared = preprocessing.prepare_model_data(data)

    # ---------------------------------------------------------------
    # PHASE 5: Train the model
    # ---------------------------------------------------------------
    model, coefficients, train_accuracy = train.train_model(
        prepared["X"],
        prepared["X_train_scaled"],
        prepared["y_train"],
        prepared["baseline"]
    )

    # ---------------------------------------------------------------
    # PHASE 6: Evaluate the model on the unseen test set
    # ---------------------------------------------------------------
    evaluate.evaluate_model(
        model,
        prepared["X_test_scaled"],
        prepared["y_test"],
        train_accuracy,
        prepared["baseline"],
        coefficients
    )

    # ---------------------------------------------------------------
    # PHASE 7: Predict survival for brand-new passengers
    # ---------------------------------------------------------------
    # The SAME scaler from Phase 4 is reused here - never re-fitted.
    predict.predict_samples(
        model,
        prepared["scaler"],
        prepared["X"]
    )


# ---------------------------------------------------------------
# The standard Python entry-point guard
# ---------------------------------------------------------------
# __name__ is a built-in variable. It equals "__main__" only when this file
# is run directly (python main.py). If another file ever imports main.py,
# __name__ would be "main" instead and the pipeline would NOT auto-run.
# This is the conventional way to make a file both runnable and importable.
if __name__ == "__main__":
    main()
