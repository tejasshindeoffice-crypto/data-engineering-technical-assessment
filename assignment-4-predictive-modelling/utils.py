"""
utils.py
========
Shared helper functions used by every other module.

These helpers exist purely to remove repetition. The project prints a lot
of formatted section headers, and duplicating the same three print() calls
in six files would be noise. Nothing here changes any behaviour - the
output produced is byte-for-byte identical to writing the prints by hand.
"""

# ---------------------------------------------------------------
# Import the pandas library
# ---------------------------------------------------------------
# 'import pandas as pd' loads the pandas library and gives it the
# short nickname 'pd'. Every pandas function is now called as pd.something()
# This nickname is a universal convention - all Python code everywhere uses 'pd'.
import pandas as pd

# 'os' is built into Python and lets us work with folders and file paths.
import os


def configure_pandas_display():
    """
    Make pandas print ALL columns instead of hiding some.

    By default, if a table is wide, pandas hides middle columns and prints "...".
    Our dataset has 28 columns, so we tell pandas: show everything.
    """
    pd.set_option("display.max_columns", None)   # None = no limit on columns shown
    pd.set_option("display.width", 200)          # allow wide lines before wrapping


def print_phase_banner(title, left_pad, right_pad):
    """
    Print the big '#' banner that announces the start of a phase.

    left_pad / right_pad are passed in explicitly so each banner keeps the
    exact spacing it had in the original single-file version.
    """
    print("\n\n" + "#" * 70)
    print("#" + " " * left_pad + title + " " * right_pad + "#")
    print("#" * 70)


def print_section(title, char="-", leading_newline=True):
    """
    Print a titled separator block, e.g.

        ----------------------------------------------------------------------
        3. AVERAGE AGE AND FARE
        ----------------------------------------------------------------------

    char           -> which character to draw the rule with ('-' or '=')
    leading_newline-> whether to print a blank line before the block
    """
    if leading_newline:
        print("\n" + char * 70)
    else:
        print(char * 70)
    print(title)
    print(char * 70)


def ensure_plots_dir(path="plots"):
    """
    Create the folder that the chart images are saved into.

    exist_ok=True stops it erroring if the folder is already there
    (so the script can be re-run safely).
    """
    os.makedirs(path, exist_ok=True)
