from langchain.tools import tool
import pandas as pd
import numpy as np
from src.tools.utils import dataset_profile_logic

@tool
def load_dataset(file_path):
    """
    Load a CSV dataset from disk into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the CSV file to read.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame with columns inferred from the CSV.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    pd.errors.EmptyDataError
        If the CSV file is empty.
    pd.errors.ParserError
        If the CSV cannot be parsed.
    """
    df = pd.read_csv(file_path)
    return df


@tool
def dataset_profile_tool(
    file_path: str,
    sample_rows: int = 5
) -> dict:
    """
    Generate a minimal, JSON-serializable profile from a CSV file path for agentic EDA.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load and profile.
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
    - Loads the dataset with pandas.read_csv and delegates profiling to the in-notebook
      dataset_profile function to avoid non-JSON argument types in tool schemas.
    - All values are cast to built-in Python types for JSON serialization.
    """
    df = pd.read_csv(file_path)
    return dataset_profile_logic(df, sample_rows)
