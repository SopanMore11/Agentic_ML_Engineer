import pandas as pd
import numpy as np


def dataset_profile_logic(
    df: pd.DataFrame,
    sample_rows: int = 5
) -> dict:
    """
    Generate a minimal, JSON-serializable profile of a pandas DataFrame for agentic EDA.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to profile.
    sample_rows : int, default 5
        Number of first rows to include as a sample in the profile.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - shape: {"rows": int, "columns": int}
        - columns: list[str]
        - dtypes: dict[str, str]
        - duplicates: {"duplicate_rows": int}
        - nulls: dict[str, {"null_count": int, "null_percentage": float}]
        - unique_values: dict[str, int]
        - numeric_summary: dict[str, {"min": float | None, "max": float | None, "mean": float | None}]
        - sample_rows: list[dict]

    Notes
    -----
    - numeric_summary is computed only for numeric columns after dropping NaNs.
    - Values are cast to built-in Python types for JSON serialization.
    """

    n_rows, n_cols = df.shape

    # Dataset-level info
    profile = {
        "shape": {
            "rows": int(n_rows),
            "columns": int(n_cols)
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicates": {
            "duplicate_rows": int(df.duplicated().sum())
        },
        "nulls": {},
        "unique_values": {},
        "numeric_summary": {},
        "sample_rows": df.head(sample_rows).to_dict(orient="records")
    }

    # Column-level stats
    for col in df.columns:
        s = df[col]

        # Nulls
        null_count = int(s.isna().sum())
        profile["nulls"][col] = {
            "null_count": null_count,
            "null_percentage": round((null_count / n_rows) * 100, 2) if n_rows else 0.0
        }

        # Uniques
        profile["unique_values"][col] = int(s.nunique(dropna=True))

        # Numeric summary (only basics)
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if not clean.empty:
                profile["numeric_summary"][col] = {
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "mean": float(clean.mean())
                }
            else:
                profile["numeric_summary"][col] = {
                    "min": None,
                    "max": None,
                    "mean": None
                }

    return profile