"""
Project-wide notebook configuration.
"""

import random
import warnings

import numpy as np
import pandas as pd

# Global random seed
RANDOM_STATE = 42

def configure_notebook():
    """Configure common notebook settings for the project."""

    # Reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)

    # Suppress unnecessary warnings
    warnings.filterwarnings("ignore")

    print("Notebook configured successfully.")