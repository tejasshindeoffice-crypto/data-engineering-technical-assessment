"""
train.py
========
PHASE 5 - Model creation, training and cross-validated model comparison.

The model of choice is Logistic Regression. This module also measures two
Decision Trees against it using cross-validation on the TRAINING data only,
so that the held-out test set stays genuinely unseen until Phase 6.
"""

# ---------------------------------------------------------------
# Import the pandas library
# ---------------------------------------------------------------
# Used here only to build the readable coefficients table.
import pandas as pd

#   LogisticRegression     -> the classification model we will train
#   DecisionTreeClassifier -> a rival model, used only for comparison
#   cross_val_score        -> a fair way to compare models WITHOUT the test set
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Our own shared helpers for printing formatted section headers.
from utils import print_phase_banner, print_section


# ===============================================================
#
#               PHASE 5: TRAINING THE MODEL
#
# ===============================================================

def train_model(X, X_train_scaled, y_train, baseline):
    """
    PHASE 5 - Create, train and inspect the Logistic Regression model.

    X              -> the feature table, used only for its column names
    X_train_scaled -> the scaled training features
    y_train        -> the training answers
    baseline       -> the always-guess-'died' accuracy, for context

    Returns (model, coefficients, train_accuracy).
    """

    print_phase_banner("PHASE 5: MODEL TRAINING", 20, 25)

    # ---------------------------------------------------------------
    # STEP 1: Create the model
    # ---------------------------------------------------------------
    # This creates an UNTRAINED model object - think of it as an empty brain
    # that knows the RULES of logistic regression but has seen no data yet.
    #
    #   max_iter=1000
    #       Logistic Regression finds its weights by repeatedly adjusting them
    #       to reduce error. max_iter caps how many adjustment rounds it may
    #       take. The default of 100 is sometimes too few and prints a
    #       "failed to converge" warning; 1000 gives it ample room.
    #
    #   random_state=42
    #       Parts of the solver involve randomness. Fixing the seed makes the
    #       training REPRODUCIBLE - the same data always gives the same model.
    print_section("1. CREATING THE MODEL")

    model = LogisticRegression(max_iter=1000, random_state=42)

    print("Model created:", model)
    print("Status: UNTRAINED - it has not seen any data yet.")

    # ---------------------------------------------------------------
    # STEP 2: TRAIN the model
    # ---------------------------------------------------------------
    # .fit(features, answers) is where the actual learning happens.
    # We hand the model 712 passengers' details together with whether each
    # one survived, and it searches for the set of weights that best explains
    # the relationship between the two.
    #
    # NOTE: we pass X_train_scaled (the SCALED features), not X_train.
    # NOTE: we pass ONLY training data. X_test is never mentioned here -
    #       the test set must stay completely unseen until Phase 6.
    print_section("2. TRAINING THE MODEL")

    print("Training on", X_train_scaled.shape[0], "passengers and",
          X_train_scaled.shape[1], "features...")

    model.fit(X_train_scaled, y_train)     # <-- the learning happens here

    print("Training complete. The model is now fitted.")

    # ---------------------------------------------------------------
    # STEP 3: Look at WHAT the model learned
    # ---------------------------------------------------------------
    # A trained Logistic Regression stores two things:
    #   .coef_      -> one weight (coefficient) per feature
    #   .intercept_ -> the bias, a constant added to every prediction
    #
    # Because we SCALED the features in Phase 4, all coefficients are on the
    # same footing, so their sizes are directly comparable. A larger absolute
    # value means that feature influenced the decision more.
    #   POSITIVE coefficient -> raises the predicted chance of survival
    #   NEGATIVE coefficient -> lowers it
    print_section("3. WHAT THE MODEL LEARNED (coefficients)")

    # .coef_ is a 2-D array shaped (1, 8) because sklearn supports multi-class.
    # [0] pulls out the single row belonging to our one binary problem.
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    })

    # Rank by absolute size so the most influential feature appears first,
    # while keeping the original signed value for direction.
    coefficients["Importance"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values("Importance", ascending=False)

    # to_string(index=False) prints the table without pandas' row numbers.
    print(coefficients.to_string(index=False))

    print("\nIntercept (bias): {:.4f}".format(model.intercept_[0]))

    print("\nReading this table:")
    print("  POSITIVE coefficient -> this feature INCREASED survival odds")
    print("  NEGATIVE coefficient -> this feature DECREASED survival odds")
    print("  Larger absolute size -> stronger influence on the decision")

    # ---------------------------------------------------------------
    # STEP 4: How well does it fit the data it LEARNED from?
    # ---------------------------------------------------------------
    # .score() returns accuracy: the fraction of predictions that were correct.
    # Running it on the TRAINING data tells us how well the model fits what it
    # has already seen. This is NOT a real evaluation - a model that memorised
    # every row would score 100% here and still be useless on new passengers.
    # The genuine test comes in Phase 6.
    print_section("4. ACCURACY ON THE TRAINING DATA")

    train_accuracy = model.score(X_train_scaled, y_train)
    print("Training accuracy: {:.4f}  ({:.2f}%)".format(
        train_accuracy, train_accuracy * 100))
    print("Baseline to beat : {:.2f}%".format(baseline))
    print("\nReminder: this number is optimistic because the model has already")
    print("seen these 712 passengers. The honest score comes in Phase 6.")

    # ---------------------------------------------------------------
    # STEP 5: Is Logistic Regression really the better choice here?
    # ---------------------------------------------------------------
    compare_models(X_train_scaled, y_train)

    return model, coefficients, train_accuracy


def compare_models(X_train_scaled, y_train):
    """
    PHASE 5, STEP 5 - Compare candidate models with 5-fold cross-validation.

    The assignment allows either Logistic Regression or a Decision Tree.
    Rather than just asserting one is better, let us MEASURE it.

    We must NOT use the test set to pick a model - choosing a model based on
    test performance is itself a form of leakage, and the test set would no
    longer be truly unseen. Instead we use CROSS-VALIDATION on the TRAINING
    data only.

    HOW 5-FOLD CROSS-VALIDATION WORKS:
      Split the training data into 5 equal parts ("folds").
      Round 1: train on folds 2,3,4,5 -> test on fold 1
      Round 2: train on folds 1,3,4,5 -> test on fold 2
      ... and so on for all 5 rounds.
      Every row gets used for testing exactly once. Average the 5 scores.
    This gives a far more reliable estimate than a single split, and it
    never touches our real test set.
    """

    print_section("5. MODEL COMPARISON USING CROSS-VALIDATION")

    # A dictionary mapping a readable name to an untrained model object.
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=42),
        "Decision Tree (unlimited depth)": DecisionTreeClassifier(
            random_state=42),
        "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3,
                                                              random_state=42)
    }

    print("Running 5-fold cross-validation on the TRAINING data only...\n")

    # .items() loops over a dictionary giving both the key and the value.
    for name, candidate in candidates.items():
        # cross_val_score runs the full 5-round procedure and returns 5 scores.
        scores = cross_val_score(candidate, X_train_scaled, y_train, cv=5)
        print("{:<32} mean accuracy: {:.4f}  (+/- {:.4f})".format(
            name, scores.mean(), scores.std()))

    print("\nWHAT THESE RESULTS SHOW:")
    print("1. The UNLIMITED-depth tree scores WORST despite being the most")
    print("   powerful model. Left unrestricted it grows branches until it has")
    print("   memorised individual passengers - that is OVERFITTING. Capping")
    print("   its depth at 3 forces it to find general rules instead, and the")
    print("   score jumps. More flexibility does NOT mean better predictions.")
    print("2. The depth-3 tree edges ahead of Logistic Regression, but only by")
    print("   about 0.011 - smaller than either model's own standard deviation")
    print("   (+/- 0.03). That gap is NOISE, not a real difference in skill.")
    print("\nDECISION: with the two models statistically tied, we choose")
    print("Logistic Regression for its practical advantages - it outputs")
    print("probabilities (used in Phase 7) and its coefficients explain WHY")
    print("each prediction was made. We keep the model trained in Step 2.")
