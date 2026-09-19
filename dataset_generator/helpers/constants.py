EXCLUDED_COLUMNS = [
    "mac",
    "window_id", 
    "window_start",
    "window_end",
    "label",
    "label_encoded",
    "label_soft",
    "phase_id",
    "assembly_id",
    "session_code"
]

def get_feature_columns(df):
    """Return feature columns (all except excluded)."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]