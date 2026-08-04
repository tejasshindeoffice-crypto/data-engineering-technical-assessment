"""
predict.py
==========
PHASE 7 - Using the trained model on brand-new, invented passengers.

This is deployment rather than evaluation: the passengers here appear
nowhere in the dataset. The module also reproduces one prediction by hand
to show that .predict() is nothing more than multiply, add, squash, threshold.
"""

# ---------------------------------------------------------------
# Import the pandas library
# ---------------------------------------------------------------
import pandas as pd

# numpy is the numerical library pandas and scikit-learn are built on.
# We use it in Phase 7 to reproduce a prediction by hand.
import numpy as np

# matplotlib is Python's plotting library. 'pyplot' is the sub-module that
# holds the actual chart commands, and 'plt' is its universal nickname.
import matplotlib.pyplot as plt

# Our own shared helpers for printing formatted section headers.
from utils import print_phase_banner, print_section


# A list of dictionaries - one dictionary per passenger.
# 'Description' is for our benefit only; it is removed before predicting.
SAMPLE_PASSENGERS = [
    {"Description": "Wealthy young woman, 1st class",
     "Age": 25, "Fare": 100.0, "Sex": 1, "SibSp": 0, "Parch": 0,
     "Pclass": 1, "Embarked_Q": 0, "Embarked_S": 0},

    {"Description": "Poor young man, 3rd class",
     "Age": 20, "Fare": 7.25, "Sex": 0, "SibSp": 0, "Parch": 0,
     "Pclass": 3, "Embarked_Q": 0, "Embarked_S": 1},

    {"Description": "Young child (boy), 3rd class, with parents",
     "Age": 5, "Fare": 15.0, "Sex": 0, "SibSp": 2, "Parch": 2,
     "Pclass": 3, "Embarked_Q": 0, "Embarked_S": 1},

    {"Description": "Elderly man, 1st class",
     "Age": 70, "Fare": 80.0, "Sex": 0, "SibSp": 0, "Parch": 0,
     "Pclass": 1, "Embarked_Q": 0, "Embarked_S": 1},

    {"Description": "Middle-aged woman, 3rd class, large family",
     "Age": 40, "Fare": 20.0, "Sex": 1, "SibSp": 3, "Parch": 3,
     "Pclass": 3, "Embarked_Q": 1, "Embarked_S": 0},

    {"Description": "Young woman, 2nd class, travelling alone",
     "Age": 28, "Fare": 26.0, "Sex": 1, "SibSp": 0, "Parch": 0,
     "Pclass": 2, "Embarked_Q": 0, "Embarked_S": 1},
]


# ===============================================================
#
#          PHASE 7: PREDICTING FOR NEW PASSENGERS
#
# ===============================================================

def predict_samples(model, scaler, X, sample_passengers=None):
    """
    PHASE 7 - Predict survival for a handful of invented passengers.

    model  -> the fitted Logistic Regression
    scaler -> the SAME StandardScaler fitted in Phase 4 (never re-fitted)
    X      -> the training feature table, used only for its column ORDER

    Returns a dictionary with the predictions and probabilities.
    """

    if sample_passengers is None:
        sample_passengers = SAMPLE_PASSENGERS

    print_phase_banner("PHASE 7: PREDICTING NEW PASSENGERS", 16, 18)

    # ---------------------------------------------------------------
    # STEP 1: Invent some passengers
    # ---------------------------------------------------------------
    # These people are NOT in the dataset. We are asking the model to judge
    # profiles it has never encountered - exactly what a deployed model does.
    #
    # CRITICAL RULE: the features must be the SAME 8 columns, in the SAME
    # ORDER, with the SAME encoding as the training data:
    #     Age, Fare, Sex, SibSp, Parch, Pclass, Embarked_Q, Embarked_S
    # The model stores one weight per POSITION, not per name. Supplying the
    # columns in a different order produces silently wrong answers with no
    # error message at all.
    print_section("1. CREATING SAMPLE PASSENGERS")

    # pd.DataFrame(list_of_dicts) turns the list into a table, using the
    # dictionary keys as column names.
    samples_df = pd.DataFrame(sample_passengers)

    # Pull the descriptions out into a separate variable - they are text and
    # must not be fed to the model.
    descriptions = samples_df["Description"]

    # *** THE MOST IMPORTANT LINE IN THIS PHASE ***
    # Selecting with X.columns guarantees the sample table has EXACTLY the
    # same columns in EXACTLY the same order as the training data.
    # Never rely on having typed the dictionary keys in the right order.
    samples_X = samples_df[X.columns]

    print("Sample passengers created:", len(samples_X))
    print("\nTheir feature values:")
    print(samples_X.to_string(index=False))

    # A safety check that the column order really does match.
    print("\nColumn order matches training data?",
          list(samples_X.columns) == list(X.columns))

    # ---------------------------------------------------------------
    # STEP 2: Scale them with the EXISTING scaler
    # ---------------------------------------------------------------
    # We call .transform() ONLY - never .fit() again.
    # The scaler already holds the mean and standard deviation learned from
    # the TRAINING data in Phase 4, and those are the exact numbers the model
    # was trained with. Re-fitting on 6 passengers would compute a completely
    # different mean and the model would misread every value.
    print_section("2. SCALING THE SAMPLES (reusing the training scaler)")

    samples_scaled = scaler.transform(samples_X)

    print("Scaled values (mean 0, std 1 in TRAINING units):")
    print(pd.DataFrame(samples_scaled, columns=X.columns).round(2).to_string(
        index=False))

    # ---------------------------------------------------------------
    # STEP 3: Predict
    # ---------------------------------------------------------------
    # .predict()       -> the final class: 0 (died) or 1 (survived)
    # .predict_proba() -> the underlying probabilities, as a 2-column array:
    #                       column 0 = probability of class 0 (died)
    #                       column 1 = probability of class 1 (survived)
    #                     The two always add up to 1.0.
    print_section("3. PREDICTIONS")

    predictions = model.predict(samples_scaled)
    probabilities = model.predict_proba(samples_scaled)

    # Build a readable results table.
    results = pd.DataFrame({
        "Passenger": descriptions,
        # A list comprehension converts 0/1 into readable words.
        "Prediction": ["SURVIVED" if p == 1 else "DIED" for p in predictions],
        # [:, 1] means "every row, column 1" -> the survival probability.
        "Survival Probability": (probabilities[:, 1] * 100).round(1)
    })

    print(results.to_string(index=False))

    # Print each result as a sentence, with a confidence label.
    print_section("DETAILED VERDICTS")

    for i in range(len(samples_X)):
        prob = probabilities[i][1] * 100      # survival probability as a %
        verdict = "SURVIVED" if predictions[i] == 1 else "DIED"

        # How far from the 50% coin-flip line is this prediction?
        # The further away, the more confident the model is.
        if abs(prob - 50) > 30:
            confidence = "very confident"
        elif abs(prob - 50) > 15:
            confidence = "fairly confident"
        else:
            confidence = "UNCERTAIN - close to a coin flip"

        print("\n{}. {}".format(i + 1, descriptions[i]))
        print("   Prediction : {}".format(verdict))
        print("   Probability: {:.1f}% survived / {:.1f}% died".format(
            prob, 100 - prob))
        print("   Confidence : {}".format(confidence))

    # ---------------------------------------------------------------
    # STEP 4: Reproduce ONE prediction by hand
    # ---------------------------------------------------------------
    explain_prediction(model, X, samples_scaled, probabilities, descriptions)

    # ---------------------------------------------------------------
    # STEP 5: Chart the sample predictions
    # ---------------------------------------------------------------
    plot_sample_predictions(descriptions, probabilities)

    return {
        "samples_X": samples_X,
        "samples_scaled": samples_scaled,
        "predictions": predictions,
        "probabilities": probabilities,
        "descriptions": descriptions,
    }


def explain_prediction(model, X, samples_scaled, probabilities, descriptions):
    """
    PHASE 7, STEP 4 - Work through passenger 1's prediction by hand.

    There is no magic inside .predict(). It is two pieces of arithmetic:
      1. a weighted sum   z = (w1*x1) + (w2*x2) + ... + intercept
      2. the sigmoid      probability = 1 / (1 + e^(-z))
    Let us compute passenger 1 manually and check it matches sklearn.
    """

    print_section("4. HOW A PREDICTION IS ACTUALLY CALCULATED")

    print("Working through passenger 1:", descriptions[0])
    print("\nStep A - multiply each SCALED feature by its learned weight:\n")

    # Take the scaled feature values for passenger 0 and the model's weights.
    passenger_scaled = samples_scaled[0]
    weights = model.coef_[0]
    intercept = model.intercept_[0]

    print("  {:<12} {:>10} {:>12} {:>14}".format(
        "Feature", "Scaled", "Weight", "Contribution"))
    print("  " + "-" * 50)

    # zip() walks several lists together, one item from each per loop.
    running_total = 0.0
    for feature, value, weight in zip(X.columns, passenger_scaled, weights):
        contribution = value * weight       # this feature's push on the decision
        running_total += contribution
        print("  {:<12} {:>10.3f} {:>12.3f} {:>14.3f}".format(
            feature, value, weight, contribution))

    print("  " + "-" * 50)
    print("  {:<12} {:>10} {:>12} {:>14.3f}".format(
        "SUM", "", "", running_total))

    # Step B: add the intercept (the model's built-in starting bias).
    z = running_total + intercept
    print("\nStep B - add the intercept (bias):")
    print("  z = {:.3f} + ({:.3f}) = {:.3f}".format(
        running_total, intercept, z))

    # Step C: push z through the sigmoid to squash it into 0..1.
    # np.exp(x) computes e to the power of x.
    probability = 1 / (1 + np.exp(-z))
    print("\nStep C - apply the sigmoid function to turn z into a probability:")
    print("  probability = 1 / (1 + e^(-z))")
    print("  probability = 1 / (1 + e^(-{:.3f})) = {:.4f}".format(z, probability))

    # Step D: apply the 0.5 decision threshold.
    print("\nStep D - apply the 0.5 decision threshold:")
    print("  {:.4f} >= 0.5  ->  predict SURVIVED".format(probability)
          if probability >= 0.5 else
          "  {:.4f} <  0.5  ->  predict DIED".format(probability))

    # Proof: our hand calculation should match sklearn exactly.
    print("\nCHECK - does this match scikit-learn?")
    print("  Calculated by hand : {:.6f}".format(probability))
    print("  sklearn's answer   : {:.6f}".format(probabilities[0][1]))
    print("  Identical?         :", round(probability, 6) ==
          round(probabilities[0][1], 6))

    print("\nThat is the whole algorithm: multiply, add, squash, threshold.")


def plot_sample_predictions(descriptions, probabilities):
    """PHASE 7, STEP 5 - Draw the sample-prediction bar chart."""

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    # Shorten the descriptions so they fit as axis labels.
    short_labels = [d[:32] for d in descriptions]
    probs_percent = probabilities[:, 1] * 100

    # Green if predicted to survive, red if not.
    bar_colours = ["#27ae60" if p >= 50 else "#c0392b" for p in probs_percent]

    ax5.barh(short_labels, probs_percent, color=bar_colours)
    # The 50% decision boundary - bars crossing it are predicted to survive.
    ax5.axvline(50, color="black", linestyle="--", linewidth=2,
                label="50% decision threshold")
    ax5.set_xlabel("Predicted Survival Probability (%)")
    ax5.set_xlim(0, 100)
    ax5.set_title("Survival Predictions for Sample Passengers",
                  fontsize=13, fontweight="bold")
    ax5.legend()

    # Write the exact percentage at the end of each bar.
    for i, p in enumerate(probs_percent):
        ax5.text(p + 1, i, "{:.1f}%".format(p), va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/sample_predictions.png", dpi=100)
    print("\nSaved: plots/sample_predictions.png")

    plt.show()
