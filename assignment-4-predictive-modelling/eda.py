"""
eda.py
======
PHASE 3 - Exploratory Data Analysis and visualisations.

Everything in this module is about UNDERSTANDING the data: descriptive
statistics, grouped survival rates, correlations and six matplotlib charts.
No modelling happens here, and nothing in this module modifies the cleaned
DataFrame it is given.
"""

# matplotlib is Python's plotting library. 'pyplot' is the sub-module that
# holds the actual chart commands, and 'plt' is its universal nickname.
import matplotlib.pyplot as plt

# Our own shared helpers for printing formatted section headers and for
# creating the output folder that the images are written into.
from utils import print_phase_banner, print_section, ensure_plots_dir


def port_name(row):
    """Convert the one-hot Embarked columns back into a readable port name."""
    if row["Embarked_Q"] == 1:
        return "Queenstown"
    elif row["Embarked_S"] == 1:
        return "Southampton"
    else:
        return "Cherbourg"


# ===============================================================
#
#            PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
#
# ===============================================================

def run_eda(data):
    """
    PHASE 3 - Explore the cleaned dataset and produce the EDA charts.

    Returns the correlation matrix, in case a caller wants to reuse it.
    """

    print_phase_banner("PHASE 3: EXPLORATORY DATA ANALYSIS", 16, 18)

    # ---------------------------------------------------------------
    # STEP 1: Build a READABLE copy just for exploring
    # ---------------------------------------------------------------
    # Our cleaned table uses codes (Sex = 0/1, and the port split into two
    # one-hot columns). Codes are perfect for a MODEL but painful for a HUMAN.
    # So for EDA only, we make a copy with human-readable label columns.
    # IMPORTANT: 'data' stays untouched - the model will still use the numbers.
    eda = data.copy()

    # .map() swaps each value using a dictionary lookup: 0 -> "Male", 1 -> "Female"
    eda["SexLabel"] = eda["Sex"].map({0: "Male", 1: "Female"})

    # Rebuild a single readable port column from the two one-hot columns.
    # Remember the encoding: Q=0 and S=0 means Cherbourg (our dropped baseline).
    # .apply(function, axis=1) runs our function once for EVERY ROW.
    #   axis=1 means "go row by row"   (axis=0 would mean "column by column")
    eda["Port"] = eda.apply(port_name, axis=1)

    # Same idea for the target, so charts read "Survived" instead of "1".
    eda["SurvivedLabel"] = eda["Survived"].map({0: "Died", 1: "Survived"})

    # ---------------------------------------------------------------
    # STEP 2: DESCRIPTIVE STATISTICS
    # ---------------------------------------------------------------
    # .describe() computes 8 summary statistics for every numeric column:
    #   count = how many non-missing values
    #   mean  = the average
    #   std   = standard deviation (how spread out the values are)
    #   min   = smallest value
    #   25%   = first quartile  (25% of values fall below this)
    #   50%   = the MEDIAN      (the middle value)
    #   75%   = third quartile
    #   max   = largest value
    # .T means TRANSPOSE - it flips rows and columns so each variable gets
    # its own row, which is far easier to read when there are many columns.
    print_section("1. DESCRIPTIVE STATISTICS")
    print(data.describe().T)

    # ---------------------------------------------------------------
    # STEP 3: SURVIVAL COUNTS
    # ---------------------------------------------------------------
    print_section("2. SURVIVAL COUNTS")

    # .value_counts() counts how many times each distinct value appears.
    # .sort_index() puts them in order 0, 1 instead of biggest-group-first.
    survival_counts = data["Survived"].value_counts().sort_index()
    print("Raw counts:")
    print("  Died     (0):", survival_counts[0])
    print("  Survived (1):", survival_counts[1])
    print("  TOTAL       :", len(data))       # len() gives the number of rows

    # normalize=True makes value_counts return PROPORTIONS (0 to 1) instead
    # of raw counts. Multiplying by 100 turns those into percentages.
    survival_percent = data["Survived"].value_counts(normalize=True) * 100
    print("\nPercentages:")
    print("  Died     : {:.2f}%".format(survival_percent[0]))
    print("  Survived : {:.2f}%".format(survival_percent[1]))

    # THE BASELINE - the single most important number in this whole phase.
    # If we simply guessed "everybody died" for every passenger, we would be
    # correct this often. Our model in Phase 6 MUST beat this to be useful.
    print("\nBASELINE ACCURACY (always guessing 'died'): {:.2f}%".format(
        survival_percent[0]))

    # ---------------------------------------------------------------
    # STEP 4: KEY AVERAGES
    # ---------------------------------------------------------------
    print_section("3. AVERAGE AGE AND FARE")

    # .mean() = the average.   .median() = the middle value when sorted.
    # We show BOTH because they disagree when the data is skewed by outliers.
    print("Average (mean) age  : {:.2f} years".format(data["Age"].mean()))
    print("Median age          : {:.2f} years".format(data["Age"].median()))
    print("Youngest passenger  : {:.2f} years".format(data["Age"].min()))
    print("Oldest passenger    : {:.2f} years".format(data["Age"].max()))

    print("\nAverage (mean) fare : {:.2f}".format(data["Fare"].mean()))
    print("Median fare         : {:.2f}".format(data["Fare"].median()))
    print("Cheapest ticket     : {:.2f}".format(data["Fare"].min()))
    print("Most expensive      : {:.2f}".format(data["Fare"].max()))

    # ---------------------------------------------------------------
    # STEP 5: GROUPED ANALYSIS - the heart of EDA
    # ---------------------------------------------------------------
    # .groupby("X")["Y"].mean() splits the table into groups by column X,
    # then computes the average of column Y inside each group.
    #
    # A CRUCIAL TRICK: 'Survived' is stored as 0 and 1, so its MEAN is exactly
    # the SURVIVAL RATE. Example: [1, 0, 1, 1] has mean 0.75 = 75% survived.
    print_section("4. SURVIVAL RATE BY GROUP")

    print("\nBy SEX:")
    survival_by_sex = eda.groupby("SexLabel")["Survived"].mean() * 100
    print(survival_by_sex.round(2))

    print("\nBy PASSENGER CLASS:")
    survival_by_class = data.groupby("Pclass")["Survived"].mean() * 100
    print(survival_by_class.round(2))

    print("\nBy PORT OF EMBARKATION:")
    survival_by_port = eda.groupby("Port")["Survived"].mean() * 100
    print(survival_by_port.round(2))

    # .agg() lets us compute SEVERAL statistics at once instead of one.
    # We ask for the mean of four different columns, split by survival outcome.
    print("\nAVERAGE CHARACTERISTICS OF SURVIVORS vs VICTIMS:")
    comparison = eda.groupby("SurvivedLabel").agg({
        "Age": "mean",       # were survivors younger?
        "Fare": "mean",      # did survivors pay more?
        "Pclass": "mean",    # were survivors in better classes?
        "SibSp": "mean"      # did survivors travel with more family?
    })
    print(comparison.round(2))

    # ---------------------------------------------------------------
    # STEP 6: CORRELATION
    # ---------------------------------------------------------------
    # .corr() measures how strongly each pair of columns moves together.
    # The result is always between -1 and +1:
    #   +1.0 = perfect POSITIVE relationship (one goes up, the other goes up)
    #    0.0 = no linear relationship at all
    #   -1.0 = perfect NEGATIVE relationship (one goes up, the other goes down)
    print_section("5. CORRELATION MATRIX")

    correlation_matrix = data.corr()
    print(correlation_matrix.round(3))

    # We usually only care about one row: how each feature relates to the TARGET.
    print("\nCORRELATION WITH SURVIVAL (strongest relationship first):")
    # .drop("Survived") removes Survived's correlation with itself (always 1.0).
    survival_corr = correlation_matrix["Survived"].drop("Survived")
    # .abs() takes absolute values, so -0.5 counts as strong as +0.5.
    # .sort_values(ascending=False) puts the strongest relationships on top.
    # .index gives us the ordering, which we then apply to the real (signed) values.
    strongest = survival_corr.abs().sort_values(ascending=False).index
    print(survival_corr[strongest].round(3))

    # ---------------------------------------------------------------
    # STEP 7: VISUALISATIONS
    # ---------------------------------------------------------------
    print_section("6. CREATING VISUALISATIONS")

    # Create a folder for the images. exist_ok=True stops it erroring if the
    # folder is already there (so the script can be re-run safely).
    ensure_plots_dir()

    # plt.subplots(rows, cols) creates ONE window holding a GRID of charts.
    #   fig  = the whole window / canvas
    #   axes = a 2x3 grid of individual chart areas, addressed as axes[row][col]
    #   figsize=(width, height) is measured in inches
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Titanic Dataset - Exploratory Data Analysis",
                 fontsize=16, fontweight="bold")

    # --- Chart 1: How many survived? (bar chart) -----------------------
    ax = axes[0][0]                      # top-left cell of the grid
    ax.bar(["Died", "Survived"],         # the labels along the x-axis
           [survival_counts[0], survival_counts[1]],   # the bar heights
           color=["#c0392b", "#27ae60"])              # red, green
    ax.set_title("Survival Counts")
    ax.set_ylabel("Number of Passengers")
    # Write the exact number on top of each bar so the chart is self-explanatory.
    for i, count in enumerate([survival_counts[0], survival_counts[1]]):
        ax.text(i, count + 10, str(count), ha="center", fontweight="bold")

    # --- Chart 2: Survival rate by sex ---------------------------------
    ax = axes[0][1]
    ax.bar(survival_by_sex.index, survival_by_sex.values,
           color=["#e91e63", "#2980b9"])
    ax.set_title("Survival Rate by Sex")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)                  # fix the y-axis 0-100 so it reads as %
    for i, rate in enumerate(survival_by_sex.values):
        ax.text(i, rate + 2, "{:.1f}%".format(rate), ha="center",
                fontweight="bold")

    # --- Chart 3: Survival rate by passenger class ---------------------
    ax = axes[0][2]
    # The x labels must be text, so we convert 1,2,3 into "1st","2nd","3rd".
    ax.bar(["1st", "2nd", "3rd"], survival_by_class.values, color="#8e44ad")
    ax.set_title("Survival Rate by Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    for i, rate in enumerate(survival_by_class.values):
        ax.text(i, rate + 2, "{:.1f}%".format(rate), ha="center",
                fontweight="bold")

    # --- Chart 4: Age distribution (histogram) -------------------------
    # A histogram splits a numeric range into "bins" and counts how many
    # values land in each bin. It shows the SHAPE of the data.
    ax = axes[1][0]
    ax.hist(data["Age"], bins=30, color="#2980b9", edgecolor="black")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Passengers")
    # axvline draws a vertical reference line at the mean.
    ax.axvline(data["Age"].mean(), color="red", linestyle="--", linewidth=2,
               label="Mean = {:.1f}".format(data["Age"].mean()))
    ax.legend()                          # show the label we just defined

    # --- Chart 5: Fare distribution (histogram) ------------------------
    ax = axes[1][1]
    ax.hist(data["Fare"], bins=40, color="#16a085", edgecolor="black")
    ax.set_title("Fare Distribution (note the long right tail)")
    ax.set_xlabel("Fare")
    ax.set_ylabel("Number of Passengers")
    ax.axvline(data["Fare"].median(), color="red", linestyle="--", linewidth=2,
               label="Median = {:.1f}".format(data["Fare"].median()))
    ax.legend()

    # --- Chart 6: Survival rate by port --------------------------------
    ax = axes[1][2]
    ax.bar(survival_by_port.index, survival_by_port.values, color="#f39c12")
    ax.set_title("Survival Rate by Port of Embarkation")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)   # tilt labels so they do not overlap
    for i, rate in enumerate(survival_by_port.values):
        ax.text(i, rate + 2, "{:.1f}%".format(rate), ha="center",
                fontweight="bold")

    # tight_layout() auto-adjusts spacing so titles and labels never overlap.
    plt.tight_layout()
    # savefig writes the image to disk. dpi=100 controls the resolution.
    plt.savefig("plots/eda_overview.png", dpi=100)
    print("Saved: plots/eda_overview.png")

    # --- Chart 7: Correlation heatmap (its own window) -----------------
    # A heatmap turns the correlation table into colours: strong positive
    # relationships in one colour, strong negative in another.
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    # imshow() draws a grid of coloured squares from a table of numbers.
    #   cmap="coolwarm" -> blue for negative, red for positive
    #   vmin/vmax pin the colour scale to -1..+1 so colours are comparable
    heatmap = ax2.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    # Put the column names on both axes.
    ax2.set_xticks(range(len(correlation_matrix.columns)))
    ax2.set_yticks(range(len(correlation_matrix.columns)))
    ax2.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax2.set_yticklabels(correlation_matrix.columns)

    # Write the actual number inside every square - colour alone is imprecise.
    # The nested loop walks every (row, column) cell of the matrix.
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]   # .iloc = select by POSITION
            ax2.text(j, i, "{:.2f}".format(value), ha="center", va="center",
                     color="white" if abs(value) > 0.5 else "black", fontsize=9)

    ax2.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    fig2.colorbar(heatmap, label="Correlation")   # the colour scale legend
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=100)
    print("Saved: plots/correlation_heatmap.png")

    # plt.show() opens the chart windows on screen.
    # NOTE: this PAUSES the script until you close the windows.
    plt.show()

    return correlation_matrix
