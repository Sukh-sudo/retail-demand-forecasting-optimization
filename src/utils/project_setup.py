"""
Project-wide notebook setup.

This module standardizes notebook behavior by:
- Configuring display settings
- Setting random seeds
- Suppressing unnecessary warnings
"""

import random
import warnings

import numpy as np
import pandas as pd

from src.utils.config import RANDOM_STATE

warnings.filterwarnings("ignore")

# Pandas display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 150)
pd.set_option("display.max_colwidth", None)

# Reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)