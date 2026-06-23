import gc
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DATA

def process_chunk(
    sales,
    calendar,
    prices,
    day_columns,
    id_columns,
):
    """
    Transform a subset of day columns into a long-format dataset
    and merge calendar and pricing information.
    """

    sales_chunk = sales[id_columns + day_columns].copy()

    sales_long = sales_chunk.melt(
        id_vars=id_columns,
        var_name="d",
        value_name="sales"
    )

    master_chunk = (
        sales_long
        .merge(calendar, on="d", how="left")
        .merge(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left"
        )
    )

    return master_chunk

def create_master_dataset(
    sales,
    calendar,
    prices,
    chunk_size=100
):
    """
    Process the sales history in manageable chunks and
    save each processed chunk as a parquet file.
    """

    chunks_dir = PROCESSED_DATA / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    id_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    day_columns = [
        c for c in sales.columns
        if c.startswith("d_")
    ]

    total_chunks = (
        len(day_columns) + chunk_size - 1
    ) // chunk_size

    for i in range(total_chunks):

        start = i * chunk_size
        end = start + chunk_size

        current_days = day_columns[start:end]

        print(
            f"Processing chunk {i+1}/{total_chunks}"
        )

        chunk = process_chunk(
            sales=sales,
            calendar=calendar,
            prices=prices,
            day_columns=current_days,
            id_columns=id_columns,
        )

        chunk.to_parquet(
            chunks_dir / f"chunk_{i+1:03}.parquet",
            index=False
        )

        del chunk
        gc.collect()

    print("\nChunk processing complete.")

def combine_chunks():
    """Combine all processed parquet chunks into a single master dataset."""

    chunks_dir = PROCESSED_DATA / "chunks"

    files = sorted(chunks_dir.glob("chunk_*.parquet"))

    if not files:
            raise FileNotFoundError("No chunk files found.")

    frames = [pd.read_parquet(f) for f in files]

    master = pd.concat(frames, ignore_index=True)

    output_path = PROCESSED_DATA / "master_dataset.parquet"
    master.to_parquet(output_path, index=False)

    print(f"Master dataset saved to: {output_path}")
    print(f"Shape: {master.shape}")

    return master