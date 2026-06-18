import warnings
import random

import numpy as np
import pandas as pd

from src.utils.config import RANDOM_STATE

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)