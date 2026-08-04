"""
evaluate.py
===========
PHASE 6 - Honest evaluation of the trained model on the held-out test set.

Produces the accuracy score, the confusion matrix, the classification
report, an overfitting check, and two charts.
"""

# matplotlib is Python's plotting library. 'pyplot' is the sub-module that
# holds the actual chart commands, and 'plt' is its universal nickname.
import matplotlib.pyplot as plt

# Evaluation tools:
#   accuracy_score        -> the fraction of predictions that were correct
#   confusion_matrix      -> a breakdown of correct vs incorrect by class
#   classification_report -> precision, recall and F1 for each class
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# Our own shared helpers for printing formatted section headers.
from utils import print_phase_banner, print_section


# ===============================================================
#
#               PHASE 6: EVALUATING THE MODEL
#
# ===============================================================

def evaluate_model(model, X_test_scaled, y_test, train_accuracy, baseline,
                   coefficients):
    """
    PHASE 6 - Measure the trained model on data it has never seen.

    model          -> the fitted Logistic Regression
    X_test_scaled  -> the scaled test features (unseen until now)
    y_test         -> the true answers for the test set
    train_accuracy -> Phase 5's training accuracy, for the overfitting check
    baseline       -> the always-guess-'died' accuracy, for context
    coefficients   -> Phase 5's coefficient table, used for the second chart

    Returns a dictionary of the headline results.
    """

    print_phase_banner("PHASE 6: MODEL EVALUATION", 19, 24)

    # ---------------------------------------------------------------
    # STEP 1: Make predictions on the UNSEEN test set
    # ---------------------------------------------------------------
    # .predict() feeds each test passenger through the trained model and
    # returns the predicted class: 0 (died) or 1 (survived).
    #
    # This is the moment the sealed test set is finally opened. These 179
    # passengers were held back in Phase 4 and the model has never seen them,
    # so their scores are an honest estimate of real-world performance.
    print_section("1. MAKING PREDICTIONS ON UNSEEN DATA")

    y_pred = model.predict(X_test_scaled)

    print("Predictions made for", len(y_pred), "unseen passengers.")

    # Show the first 15 predictions next to the true answers so you can see
    # what is actually being compared.
    print("\nFirst 15 predictions vs the real outcomes:")
    print("  Predicted:", y_pred[:15].tolist())
    # .values converts the pandas Series into a plain array for a fair comparison.
    print("  Actual   :", y_test.values[:15].tolist())

    # Comparing two arrays with == gives True/False for each position.
    matches = (y_pred[:15] == y_test.values[:15])
    # A list comprehension turns those booleans into readable tick/cross marks.
    print("  Correct? :", ["OK" if m else "XX" for m in matches])

    # ---------------------------------------------------------------
    # STEP 2: ACCURACY SCORE
    # ---------------------------------------------------------------
    # Accuracy = (number of correct predictions) / (total predictions)
    print_section("2. ACCURACY SCORE")

    accuracy = accuracy_score(y_test, y_pred)

    print("TEST ACCURACY: {:.4f}  ({:.2f}%)".format(accuracy, accuracy * 100))

    # Convert the fraction back into a count of passengers, which is easier
    # to picture than a percentage. round() avoids floating point ugliness.
    correct = round(accuracy * len(y_test))
    print("The model got {} of {} passengers right ({} wrong).".format(
        correct, len(y_test), len(y_test) - correct))

    # CONTEXT IS EVERYTHING - a raw accuracy number means nothing on its own.
    print("\nPutting that number in context:")
    print("  Baseline (always guess 'died') : {:.2f}%".format(baseline))
    print("  Our model on UNSEEN data       : {:.2f}%".format(accuracy * 100))
    print("  Improvement over baseline      : {:.2f} percentage points".format(
        accuracy * 100 - baseline))

    # OVERFITTING CHECK - compare performance on seen vs unseen data.
    print("\nOverfitting check:")
    print("  Accuracy on TRAINING data (seen)   : {:.2f}%".format(
        train_accuracy * 100))
    print("  Accuracy on TEST data (unseen)     : {:.2f}%".format(
        accuracy * 100))
    gap = (train_accuracy - accuracy) * 100
    print("  Gap between them                   : {:.2f} percentage points".format(gap))

    # A small gap means the model generalises; a large gap means it memorised.
    if abs(gap) < 5:
        print("  VERDICT: small gap -> the model GENERALISES well, no overfitting.")
    else:
        print("  VERDICT: large gap -> the model may be OVERFITTING.")

    # ---------------------------------------------------------------
    # STEP 3: CONFUSION MATRIX
    # ---------------------------------------------------------------
    # Accuracy hides WHICH kind of mistake was made. The confusion matrix
    # breaks the results into four groups:
    #
    #                          PREDICTED
    #                      Died      Survived
    #   ACTUAL  Died    [   TN    |     FP    ]
    #           Surv    [   FN    |     TP    ]
    #
    #   TN = True  Negative -> said died,     really died      (correct)
    #   FP = False Positive -> said survived, really died      (wrong)
    #   FN = False Negative -> said died,     really survived  (wrong)
    #   TP = True  Positive -> said survived, really survived  (correct)
    #
    # The DIAGONAL (TN and TP) holds the correct predictions.
    print_section("3. CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred)

    print("Raw matrix:")
    print(cm)

    # .ravel() flattens the 2x2 matrix into 4 values in reading order,
    # which we unpack into four clearly named variables.
    tn, fp, fn, tp = cm.ravel()

    print("\nLabelled version:")
    print("                        PREDICTED")
    print("                   Died      Survived")
    print("  ACTUAL Died    {:>6}    {:>8}".format(tn, fp))
    print("         Surv    {:>6}    {:>8}".format(fn, tp))

    print("\nWhat each number means:")
    print("  True  Negatives (TN) = {:>3}  -> said DIED,     really DIED      CORRECT".format(tn))
    print("  False Positives (FP) = {:>3}  -> said SURVIVED, really DIED      wrong".format(fp))
    print("  False Negatives (FN) = {:>3}  -> said DIED,     really SURVIVED  wrong".format(fn))
    print("  True  Positives (TP) = {:>3}  -> said SURVIVED, really SURVIVED  CORRECT".format(tp))

    print("\n  Total correct : {} + {} = {}".format(tn, tp, tn + tp))
    print("  Total wrong   : {} + {} = {}".format(fp, fn, fp + fn))
    print("  Accuracy      : {} / {} = {:.4f}   (matches Step 2)".format(
        tn + tp, len(y_test), (tn + tp) / len(y_test)))

    # ---------------------------------------------------------------
    # STEP 4: CLASSIFICATION REPORT
    # ---------------------------------------------------------------
    # Three metrics, each answering a different question:
    #
    #   PRECISION = TP / (TP + FP)
    #       "When the model says SURVIVED, how often is it right?"
    #       Punishes false alarms.
    #
    #   RECALL    = TP / (TP + FN)
    #       "Of all the passengers who really survived, how many did we find?"
    #       Punishes missed cases.
    #
    #   F1-SCORE  = the harmonic mean of precision and recall
    #       A single balanced score. It is only high when BOTH are high.
    #
    #   SUPPORT   = how many real passengers of that class are in the test set.
    print_section("4. CLASSIFICATION REPORT")

    # target_names replaces the labels 0 and 1 with readable words.
    # digits=3 shows three decimal places.
    report = classification_report(y_test, y_pred,
                                   target_names=["Died (0)", "Survived (1)"],
                                   digits=3)
    print(report)

    # Now compute the same numbers by hand, to prove where they come from.
    print("Verifying the 'Survived' row by hand from the confusion matrix:")
    print("  Precision = TP / (TP + FP) = {} / ({} + {}) = {:.3f}".format(
        tp, tp, fp, tp / (tp + fp)))
    print("  Recall    = TP / (TP + FN) = {} / ({} + {}) = {:.3f}".format(
        tp, tp, fn, tp / (tp + fn)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("  F1        = 2 * (P * R) / (P + R)      = {:.3f}".format(f1))

    # ---------------------------------------------------------------
    # STEP 5: VISUALISE THE CONFUSION MATRIX
    # ---------------------------------------------------------------
    print_section("5. VISUALISING THE RESULTS")

    fig3, ax3 = plt.subplots(figsize=(7, 6))

    # imshow draws the 2x2 grid of numbers as coloured squares.
    im = ax3.imshow(cm, cmap="Blues")

    # Label both axes with readable class names.
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Predicted: Died", "Predicted: Survived"])
    ax3.set_yticklabels(["Actual: Died", "Actual: Survived"])

    # Write the count AND its meaning inside each of the four cells.
    labels = [["True Negative", "False Positive"],
              ["False Negative", "True Positive"]]

    for i in range(2):
        for j in range(2):
            # Dark cells need white text, light cells need black text.
            text_colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax3.text(j, i, "{}\n{}".format(cm[i, j], labels[i][j]),
                     ha="center", va="center", color=text_colour,
                     fontsize=13, fontweight="bold")

    ax3.set_title("Confusion Matrix - Test Set ({} passengers)\nAccuracy: {:.2f}%"
                  .format(len(y_test), accuracy * 100),
                  fontsize=13, fontweight="bold")
    fig3.colorbar(im, label="Number of Passengers")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=100)
    print("Saved: plots/confusion_matrix.png")

    # --- A second chart: feature importance from the coefficients -------
    fig4, ax4 = plt.subplots(figsize=(9, 6))

    # Sort so the chart reads cleanly from most negative to most positive.
    coef_sorted = coefficients.sort_values("Coefficient")

    # Green bars raise survival odds, red bars lower them.
    colours = ["#c0392b" if c < 0 else "#27ae60"
               for c in coef_sorted["Coefficient"]]

    # barh() draws HORIZONTAL bars, which suits long feature names.
    ax4.barh(coef_sorted["Feature"], coef_sorted["Coefficient"], color=colours)
    ax4.axvline(0, color="black", linewidth=0.8)   # a reference line at zero
    ax4.set_xlabel("Coefficient (effect on survival odds)")
    ax4.set_title("What the Model Learned - Feature Influence",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=100)
    print("Saved: plots/feature_importance.png")

    plt.show()

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
