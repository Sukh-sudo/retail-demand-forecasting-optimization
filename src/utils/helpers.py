import pandas as pd


def missing_values(df):

    return (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )


def dataset_summary(df):

    return df.describe(include="all").T


def memory_usage(df):

    return (
        df.memory_usage(deep=True).sum()
        / 1024**2
    )


def save_parquet(df, path):

    df.to_parquet(path, index=False)