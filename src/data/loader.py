import pandas as pd

from src.utils.config import RAW_DATA


def load_data():

    calendar = pd.read_csv(RAW_DATA / "calendar.csv")

    sales = pd.read_csv(
        RAW_DATA / "sales_train_validation.csv"
    )

    prices = pd.read_csv(
        RAW_DATA / "sell_prices.csv"
    )

    return calendar, sales, prices