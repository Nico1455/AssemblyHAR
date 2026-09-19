"""
Ensemble feature selection module with saving.

The module performs:
- Optional Kernel Discriminant Analysis (KDA) transformation
- Mutual Information
- ANOVA F-test
- RandomForest importance
- Recursive Feature Elimination (RFE)

All importance scores are normalized and averaged into a single ensemble score.
The top features that together contribute 90% of total ensemble importance are selected.
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from math import floor
from .constants import EXCLUDED_COLUMNS, get_feature_columns
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import rbf_kernel


def fit_kda(
    df: pd.DataFrame,
    results_folder: str,
    n_components: int = None,
    gamma: float = None
):
    """
    Fit Kernel Discriminant Analysis (KDA) on training data.
    
    KDA finds non-linear projections that maximize between-class variance
    and minimize within-class variance using the kernel trick (RBF kernel).
    
    This function should be called on TRAINING data only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training windowed feature dataframe with 'label_encoded' column.
    results_folder : str
        Folder to save the fitted KDA transformer.
    n_components : int, optional
        Number of KDA components to keep (default: n_classes - 1).
    gamma : float, optional
        RBF kernel gamma parameter (default: 1 / n_features).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Fitted KDA model containing X_train, alpha, gamma, and other params.
    """
    
    if "label_encoded" not in df.columns:
        raise ValueError("DataFrame must contain 'label_encoded' column for KDA.")
    
    # Separate features from metadata
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["label_encoded"].values
    
    # Sanity check for NaN/inf (should be cleaned before calling)
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(
            "Training data contains NaN or inf values. "
            "Please clean the data before applying KDA transformation."
        )
    
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    
    # Set defaults
    if n_components is None:
        n_components = n_classes - 1
    if gamma is None:
        gamma = 1.0 / n_features
    
    print(f"\n[KDA] Fitting KDA on {len(X)} training samples with {n_features} features...")
    print(f"[KDA] RBF kernel gamma={gamma:.6f}, target components: {n_components} (from {n_classes} classes)")
    
    # Step 1: Compute RBF kernel matrix
    K = rbf_kernel(X, X, gamma=gamma)
    
    # Step 2: Compute within-class kernel matrix W
    W = np.zeros((len(X), len(X)))
    for k in range(n_classes):
        idx_k = np.where(y == k)[0]
        n_k = len(idx_k)
        if n_k > 0:
            for i in idx_k:
                for j in idx_k:
                    W[i, j] = 1.0 / n_k
    
    # Step 3: Solve generalized eigenvalue problem: K @ W @ K @ alpha = lambda @ K @ K @ alpha
    KWK = K @ W @ K
    KK = K @ K
    
    # Add small regularization for numerical stability
    KK += 1e-6 * np.eye(len(KK))
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(KK) @ KWK)
    except np.linalg.LinAlgError:
        warnings.warn("[KDA] Eigendecomposition failed, using pseudo-inverse")
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(KK) @ KWK)
    
    # Step 4: Sort by largest eigenvalues and select top n_components
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Keep only n_components
    n_components = min(n_components, len(eigenvalues))
    alpha = eigenvectors[:, :n_components]
    
    print(f"[KDA] Selected {n_components} components with eigenvalues: {eigenvalues[:n_components]}")
    
    # Store KDA transformer
    kda_model = {
        "X_train": X,  # Store training samples for kernel computation
        "alpha": alpha,
        "gamma": gamma,
        "n_components": n_components,
        "feature_cols": feature_cols,
        "kda_feature_names": [f"KDA_{i+1}" for i in range(n_components)]
    }
    
    # Save to disk
    kda_path = Path(results_folder) / "kda_transformer.pkl"
    with open(kda_path, "wb") as f:
        pickle.dump(kda_model, f)
    print(f"[KDA] Saved KDA transformer to {kda_path.resolve()}")
    
    return kda_model


def transform_kda(df: pd.DataFrame, kda_model: dict):
    """
    Apply fitted KDA transformation to new data (train/val/test).
    
    Parameters
    ----------
    df : pd.DataFrame
        Windowed feature dataframe to transform.
    kda_model : dict
        Fitted KDA model from fit_kda().
    
    Returns
    -------
    pd.DataFrame
        DataFrame with KDA-transformed features, metadata columns preserved.
    """
    
    # Extract model parameters
    X_train = kda_model["X_train"]
    alpha = kda_model["alpha"]
    gamma = kda_model["gamma"]
    feature_cols = kda_model["feature_cols"]
    kda_feature_names = kda_model["kda_feature_names"]
    
    # Separate features from metadata
    X = df[feature_cols].values
    
    # Sanity check for NaN/inf (should be cleaned before calling, but handle if not)
    if np.isnan(X).any() or np.isinf(X).any():
        print(f"[KDA Transform] Warning: Found NaN/inf values in transform data, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute kernel between new data and training data
    K_new = rbf_kernel(X, X_train, gamma=gamma)
    
    # Project onto KDA components
    X_kda = K_new @ alpha
    
    # Create new dataframe with KDA features
    kda_df = pd.DataFrame(X_kda, columns=kda_feature_names, index=df.index)
    
    # Preserve metadata columns
    metadata_cols = [col for col in df.columns if col not in feature_cols]
    result_df = pd.concat([df[metadata_cols].reset_index(drop=True), kda_df.reset_index(drop=True)], axis=1)
    
    return result_df


def select_features_ensemble(
    df: pd.DataFrame,
    results_folder: str,
    window_size_sec: float,
    step_size_sec: float,
    phase_totals: pd.DataFrame = None,
    assembly_totals: pd.DataFrame = None,
    activity_totals: pd.DataFrame = None,
    activity_segments: pd.DataFrame = None,
    cumulative_threshold: float = 0.90,
    random_state: int = 42
):
    """
    Perform ensemble feature selection and save results.

    Parameters
    ----------
    df : pd.DataFrame
        Windowed feature dataframe.
    results_folder : str
        The folder where feature_ranking.csv and selected_features.txt are saved.
    label_column : str
        Column containing labels.
    cumulative_threshold : float
        Percentage (0–1) of cumulative ensemble importance to keep.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    reduced_df : pd.DataFrame
        Same dataframe but only with selected feature columns (plus ID columns).
    """

    os.makedirs(results_folder, exist_ok=True)
    label_column: str = "label_encoded"

    # ====== Get only the feature columns ======
    feature_cols = get_feature_columns(df)

    # ====== Remove NaN rows (only in features) ======
    before = len(df)
    df = df.dropna(subset=feature_cols)
    after = len(df)
    print(f"Removed {before - after} rows containing NaN values in feature columns.")

    # ====== Compute and save class counts as well as phase and assembly totals ======
    class_counts = df[label_column].value_counts().sort_index()

    # Build encoded → string mapping
    mapping = (
        df[[label_column, "label"]]
        .drop_duplicates()
        .set_index(label_column)["label"]
    )

    # Base counts df
    class_counts_df = (
        class_counts.to_frame(name="num_windows") 
        .join(mapping)
        .reset_index()
        .rename(columns={
            label_column: "encoded_label",
            "label": "label"
        })
    )

    # Ensure activity_totals is provided
    if activity_totals is None:
        raise ValueError(
            "select_features_ensemble() was called without activity_totals. "
            "This is required for duration-based statistics."
        )

    # Ensure correct column names exist
    required_cols = {"label", "total_duration_sec", "total_repetitions", "avg_duration_sec"}
    if not required_cols.issubset(activity_totals.columns):
        raise ValueError(
            f"activity_totals must contain columns: {required_cols}"
        )

    # --- Safety check: detect mismatches before merging ---
    left_labels = set(class_counts_df["label"])
    right_labels = set(activity_totals["label"])
    missing_in_totals = left_labels - right_labels
    missing_in_counts = right_labels - left_labels
    if missing_in_totals:
        print("\n WARNING: Labels in class_counts_df but not in activity_totals:")
        for label in sorted(missing_in_totals):
            print(f"  - {label}")
    if missing_in_counts:
        print("\n Note: Labels in activity_totals but not in class_counts_df:")
        for label in sorted(missing_in_counts):
            print(f"  - {label}")

    # Merge class counts with activity total durations, average duration, and repetitions
    class_counts_df = class_counts_df.merge(
        activity_totals[["label", "total_duration_sec", "total_repetitions", "avg_duration_sec"]],
        left_on="label",
        right_on="label",
        how="left"
    )

    # Compute expected number of windows for each activity from activity segments
    if activity_segments is None or activity_segments.empty:
        raise ValueError(
            "select_features_ensemble() requires 'activity_segments' parameter. "
        )
    
    # Ensure datetime columns
    if "start_time" in activity_segments.columns:
        activity_segments["start_time"] = pd.to_datetime(activity_segments["start_time"])
    if "end_time" in activity_segments.columns:
        activity_segments["end_time"] = pd.to_datetime(activity_segments["end_time"])
    # Ensure duration column exists
    if "duration" not in activity_segments.columns and "start_time" in activity_segments.columns and "end_time" in activity_segments.columns:
        activity_segments["duration"] = (activity_segments["end_time"] - activity_segments["start_time"]).dt.total_seconds()
    
    # Function to compute windows from duration using outer window/step variables
    def segment_windows_from_duration(duration, eps: float = 1e-9):
        if pd.isna(duration) or duration < window_size_sec:
            return 0
        # number of additional steps after the first full window
        n = floor((duration - window_size_sec) / step_size_sec + eps) + 1
        return max(0, int(n))
    
    # Compute windows per segment
    activity_segments["expected_windows_seg"] = activity_segments["duration"].apply(segment_windows_from_duration)
    
    # Sum windows per activity label and multiply by number of macs
    expected_windows_by_label = (
        activity_segments.groupby("label")["expected_windows_seg"]
        .sum()
        .to_dict()
    )
    
    #n_macs = df["mac"].nunique()
    #for k in expected_windows_by_label:
    #    expected_windows_by_label[k] *= n_macs
    
    class_counts_df["expected_windows"] = class_counts_df["label"].map(
        lambda lab: expected_windows_by_label.get(lab, 0)
    )

    # Reorder columns
    class_counts_df = class_counts_df[
        ["encoded_label", "label", "total_repetitions", "total_duration_sec",
        "avg_duration_sec", "expected_windows", "num_windows"]
    ]

    print("Class counts:")
    print(class_counts_df)

    # ====== Extract X, y ======
    X = df[feature_cols].values
    y = df[label_column].values

    # ====== 1. Mutual Information ======
    mi = mutual_info_classif(X, y, random_state=random_state)
    mi = MinMaxScaler().fit_transform(mi.reshape(-1, 1)).flatten()

    # ====== 2. ANOVA F-test ======
    f_values, _ = f_classif(X, y)
    f_values = MinMaxScaler().fit_transform(f_values.reshape(-1, 1)).flatten()

    # ====== 3. RandomForest Importance ======
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X, y)
    rf_imp = MinMaxScaler().fit_transform(rf.feature_importances_.reshape(-1, 1)).flatten()

    # ====== 4. Recursive Feature Elimination (RFE) - DISABLED FOR SPEED ======
    # RFE is slow (requires sequential model training for each feature removal)

    # warnings.filterwarnings(
    # "ignore",
    # message="Using the 'liblinear' solver for multiclass classification is deprecated"
    # )
    #
    # lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
    # rfe = RFE(lr, n_features_to_select=1)
    # rfe.fit(X, y)
    #
    # # Higher = better → invert ranking
    # rfe_rank = (len(feature_cols) - rfe.ranking_) + 1
    # rfe_rank = MinMaxScaler().fit_transform(rfe_rank.reshape(-1, 1)).flatten()

    # ====== Combine into ensemble score ======
    # Currently using MI + ANOVA + RandomForest (uncomment rfe_rank and change divisor to 4.0 to include RFE)
    ensemble = (mi + f_values + rf_imp) / 3.0

    ranking_df = pd.DataFrame({
        "feature": feature_cols,
        "mutual_information": mi,
        "anova_f": f_values,
        "rf_importance": rf_imp,
        # "rfe_importance": rfe_rank,  # Uncomment if RFE is re-enabled
        "ensemble_importance": ensemble
    })

    ranking_df = ranking_df.sort_values("ensemble_importance", ascending=False)
    ranking_df.reset_index(drop=True, inplace=True)

    # ====== Compute cumulative importance ======
    total = ranking_df["ensemble_importance"].sum()
    ranking_df["cumulative_importance"] = ranking_df["ensemble_importance"].cumsum() / total

    # ====== Select top 90% ======
    cum = ranking_df["cumulative_importance"].values  

    # Index of the first entry >= threshold
    cutoff_idx = np.searchsorted(cum, cumulative_threshold, side="right")

    # Select all features up to that index
    selected_features = ranking_df.iloc[:cutoff_idx]["feature"].tolist()


    # ====== Save outputs ======
    ranking_path = os.path.join(results_folder, "feature_ranking.csv")
    list_path = os.path.join(results_folder, "selected_features.txt")

    ranking_df.to_csv(ranking_path, index=False)

    with open(list_path, "w") as f:
        for feat in selected_features:
            f.write(feat + "\n")

    # ====== Return reduced dataframe ======
    # Keep non-feature columns + selected features
    keep_cols = EXCLUDED_COLUMNS + selected_features
    existing_keep_cols = [c for c in keep_cols if c in df.columns]

    reduced_df = df[existing_keep_cols].copy()

    return reduced_df, selected_features
