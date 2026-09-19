import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import os


def get_results_subfolder(data_file: Path, session_code: str, order_condition: str) -> Path:
    """Returns the session results directory and creates it if missing."""
    results_folder = data_file.parent / f"session_{session_code}_{order_condition}_results"
    results_folder.mkdir(exist_ok=True)
    return results_folder


def save_dataframe(df: pd.DataFrame, output_folder: Path, filename: str):
    """Saves a dataframe to a CSV file inside output_folder.

    Converts array-like objects to JSON strings to avoid pandas Series dumps in CSV.
    """
    outfile = output_folder / f"{filename}.csv"

    def _serialize_cell(x):
        # Handle pandas Series first (before checking type)
        if isinstance(x, pd.Series):
            return json.dumps(x.tolist())
        # Handle numpy arrays
        if isinstance(x, np.ndarray):
            return json.dumps(x.tolist())
        # Handle plain lists/tuples
        if isinstance(x, (list, tuple)):
            return json.dumps(list(x))
        # Handle dictionaries (e.g., soft-label distributions)
        if isinstance(x, dict):
            return json.dumps(x)
        return x

    df_to_save = df.copy()
    
    # Iterate through all columns and sanitize object-type values
    for col in df_to_save.columns:
        # Check if column contains objects that might be Series/arrays
        if df_to_save[col].dtype == 'object':
            df_to_save[col] = df_to_save[col].apply(_serialize_cell)

    df_to_save.to_csv(outfile, index=False)
    # print(f"Saved dataframe to: {outfile.resolve()}")


def save_text(text: str, results_folder: Path, filename: str):
    """Saves plain text to a TXT file."""
    outfile = results_folder / f"{filename}.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(text)
    # print(f"Saved text to: {outfile.resolve()}")


def parse_session_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Parse session folder name to extract session_code and order_condition.

    Expected format: "session_<code>_<condition>_data" or "session_<code>_<condition>"
    
    Parameters
    ----------
    folder_name : str
        Folder name (e.g., "session_1_O1_data")

    Returns
    -------
    tuple[str, str]
        (session_code, order_condition) or raises ValueError if format is invalid
    """
    if not folder_name.startswith("session_"):
        raise ValueError(f"Invalid session folder name: {folder_name}")
    
    parts = folder_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid session folder name format: {folder_name}")
    
    session_code = parts[1]
    order_condition = parts[2]
    
    if order_condition not in ("O1", "O2"):
        raise ValueError(f"Unrecognized order_condition: {order_condition}")
    
    return session_code, order_condition


def load_session_result(session_folder: Path, session_code: str, order_condition: str, result_name: str, parse_dates: List[str] | None = None) -> pd.DataFrame | None:
    """Load a single session result CSV.

    Returns None if the file does not exist.
    """
    results_sub = session_folder / f"session_{session_code}_{order_condition}_results"
    file_path = results_sub / f"session_{session_code}_{order_condition}_{result_name}.csv"
    if not file_path.exists():
        return None
    return pd.read_csv(file_path, parse_dates=parse_dates)


def concat_session_results(dfs: List[pd.DataFrame], *,
                           required_cols: List[str] | None = None,
                           parse_dates: List[str] | None = None,
                           compute_duration: bool = False,
                           start_col: str = "start_time",
                           end_col: str = "end_time",
                           duration_col: str = "duration",
                        ) -> pd.DataFrame:
    """
    Standardize and concatenate per-session DataFrames.

    - `dfs` is a list of DataFrames (some may be None or empty).
    - `required_cols` (optional): if provided, DataFrames missing these columns are skipped.
    - `parse_dates`: list of column names to parse as datetimes (applied if columns exist).
    - `compute_duration`: if True and `duration_col` not present, compute it as (end - start).dt.total_seconds().

    Returns a concatenated DataFrame (or empty DataFrame if no valid inputs).
    """
    valid = [df.copy() for df in dfs if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame()

    # Parse date columns if requested
    if parse_dates:
        for df in valid:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

    # Compute duration if requested
    if compute_duration:
        for df in valid:
            if duration_col not in df.columns and start_col in df.columns and end_col in df.columns:
                df[duration_col] = (df[end_col] - df[start_col]).dt.total_seconds()

    # Enforce required columns if provided
    if required_cols:
        cleaned = []
        for df in valid:
            if all((c in df.columns) for c in required_cols):
                cleaned.append(df)
        valid = cleaned
        if not valid:
            return pd.DataFrame()

    out = pd.concat(valid, ignore_index=True, sort=False)
    return out

def _parse_json_strings_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse JSON-formatted strings back to Python lists in object columns.
    
    When DataFrames with lists are saved to CSV, lists are converted to JSON strings.
    This function reverses that process when loading the CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing JSON-formatted strings
    
    Returns
    -------
    pd.DataFrame
        DataFrame with JSON strings parsed back to lists
    """
    def parse_if_json(x):
        if isinstance(x, str) and x.strip() and x.strip()[0] in {'[', '{'}:
            try:
                return json.loads(x)
            except (json.JSONDecodeError, ValueError):
                return x
        return x
    
    df_parsed = df.copy()
    for col in df_parsed.columns:
        if df_parsed[col].dtype == 'object':
            df_parsed[col] = df_parsed[col].apply(parse_if_json)
    return df_parsed


def load_session_windowed_df(session_folder: Path, session_code: str, order_condition: str, data_key: str) -> pd.DataFrame:
    """
    Load an already-computed session windowed dataframe from disk.

    Parameters
    ----------
    session_folder : Path
        Folder that contains the preprocessed session CSVs.
    session_code : str
        Identifier for the session.
    order_condition : str
        The experimental order condition.

    Returns
    -------
    pd.DataFrame
        The windowed dataframe including a `session_code` column.
        `window_start` and `window_end` are parsed as datetime when possible.

    Raises
    ------
    FileNotFoundError
        If the expected CSV does not exist.
    """
    file_path = session_folder / f"session_{session_code}_{order_condition}_windowed_{data_key}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Windowed {data_key} file not found: {file_path.resolve()}")
    df = pd.read_csv(file_path)

    # Parse JSON strings back to Python objects (lists/dicts)
    df = _parse_json_strings_in_df(df)
    
    # Ensure datetimes for window_start/window_end if present
    for col in ("window_start", "window_end"):
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                # keep as-is if parse fails
                pass
    # add session provenance column
    df["session_code"] = session_code
    return df


def combine_activity_totals(all_activity_totals: list[pd.DataFrame], activity_segments_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Combine per-session activity_totals DataFrames and compute total durations,
    total repetitions, and average duration per repetition per activity.

    Parameters
    ----------
    all_activity_totals : list[pd.DataFrame]
        List of per-session activity_totals DataFrames.
    activity_segments_df : pd.DataFrame, optional
        Combined activity segments from all sessions. If provided, used to compute
        std across individual segment durations instead of per-session totals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - activity_id
            - label
            - total_duration_sec  (summed across all sessions)
            - total_repetitions   (summed across all sessions)
            - avg_duration_sec    (total_duration_sec / total_repetitions)
            - std_duration_sec    (std across individual segment durations if activity_segments_df provided)
    """
    if len(all_activity_totals) == 0:
        raise ValueError("No activity_totals DataFrames found.")

    # 1) Concatenate all per-session totals
    df_all = pd.concat(all_activity_totals, ignore_index=True)

    # 2) Sum durations and repetitions per activity
    df_global = (
        df_all.groupby(["activity_id", "label"])
              .agg(
                  total_duration_sec=("total_duration_sec", "sum"),
                  total_repetitions=("repetitions", "sum")
              )
              .reset_index()
    )

    # 3) Compute std across individual activity segment durations if available
    if activity_segments_df is not None:
        std_activity = (
            activity_segments_df
            .groupby(["activity_id", "label"])["duration"].std()
            .reset_index()
            .rename(columns={"duration": "std_duration_sec"})
        )
        df_global = df_global.merge(std_activity, on=["activity_id", "label"], how="left")

    # 4) Compute average duration per repetition
    df_global["avg_duration_sec"] = df_global["total_duration_sec"] / df_global["total_repetitions"]

    # Sort by activity_id for consistency
    df_global = df_global.sort_values("activity_id").reset_index(drop=True)

    return df_global


# Dataset handling that overwrites existing datasets
def write_dataset_overwrite(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, dataset_root: Path):
    """
    Save train/validation/test dataframes to disk, replacing any existing files.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split.
    val_df : pd.DataFrame
        Validation split.
    test_df : pd.DataFrame
        Test split.
    dataset_root : Path
        Folder where the dataset CSVs will be written.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    save_dataframe(train_df, dataset_root, "train")
    save_dataframe(val_df, dataset_root, "val")
    save_dataframe(test_df, dataset_root, "test")
    print(f"Wrote/overwrote dataset splits to {dataset_root}")


def setup_logger(log_dir: str, log_name: str = "dataset_generation.log") -> logging.Logger:
    """
    Setup a logger that writes both to console and a log file.

    Parameters
    ----------
    log_dir : str
        Directory where log file is written.
    log_name : str
        Log filename.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("dataset_generation")
    logger.setLevel(logging.INFO)

    # Avoid duplicate console handlers
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(console)

    # Remove old file handlers before adding new one
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    log_path = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_path}")
    return logger


def get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """
    Return a usable logger instance.

    If an external logger is provided, it is returned unchanged.
    If None is passed, a module-level logger (using this module's __name__)
    is created or retrieved via logging.getLogger(__name__).

    This allows all functions to use consistent logging behavior without
    requiring callers to supply a logger explicitly.

    Args:
        logger: A logger instance provided by the caller, or None.

    Returns:
        A valid logging.Logger object.
    """
    return logger if logger is not None else logging.getLogger(__name__)
