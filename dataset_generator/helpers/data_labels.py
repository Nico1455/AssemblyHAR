import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Activity configuration
# ---------------------------------------------------------------------------
MOTOR1_ACTIVITIES = [
    "Pick up motor housing",
    "Insert rotor",
    "Attach brushes",
    "Connect wires",
    "Attach power source",
    "Attach end cap",
    "Tighten cover",
    "Place in package",
    "Close package"
]

MOTOR2_ACTIVITIES = [
    "Pick up rotor",
    "Attach fan",
    "Attach bearings",
    "Set rotor aside",
    "Retrieve lower housing half",
    "Insert stator",
    "Close housing",
    "Attach terminal box",
    "Insert rotor unit",
    "Place cover"
]


# -------------------------------------------------------------
# Parse event strings into categories
# -------------------------------------------------------------
def parse_event_text(text: str) -> pd.Series:
    """
    Parse a raw event string into structured components.

    Supported formats:
    - "Phase start M1: 3"
    - "Assembly start M2: 1"
    - "Activity end 3 - Insert rotor"
    - "Last Assembly Activity ended"

    Returns
    -------
    pd.Series:
        event_type : str
        motor_logged : str or None
        phase_id_logged : int or None
        activity_id_logged : int or None
        activity_label_logged : str or None
    """
    text = str(text)

    evt_type = text
    motor = None
    phase_id = None
    act_id = None
    act_label = None

    if text.startswith("Phase start"):
        evt_type = "Phase start"
        cleaned = text.split(":")[0].strip()
        parts = cleaned.split()
        # expected "Phase start M1"
        if len(parts) >= 3:
            motor = parts[2]
        # extract ": N"
        if ":" in text:
            try:
                phase_id = int(text.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    elif text.startswith("Assembly start"):
        evt_type = "Assembly start"
        cleaned = text.split(":")[0].strip()
        parts = cleaned.split()
        if len(parts) >= 3:
            motor = parts[2]

    elif text.startswith("Activity end"):
        evt_type = "Activity end"
        try:
            after = text.replace("Activity end", "").strip()
            act_id = int(after.split("-")[0].strip())
            act_label = after.split("-", 1)[1].strip()
        except (ValueError, IndexError):
            pass

    elif text.startswith("Last Assembly Activity ended"):
        evt_type = "Last Assembly Activity ended"

    return pd.Series([evt_type, motor, phase_id, act_id, act_label])


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------
def label_accelerometer_data(
    df: pd.DataFrame,
    button_log: pd.DataFrame,
    order_condition: str
) -> pd.DataFrame:
    """
    Label accelerometer samples using button-press event logs.

    Parameters
    ----------
    df : pd.DataFrame
        Accelerometer samples with at least:
        - abs_time: timestamp
        - mac: device identifier
    button_log : pd.DataFrame
        Event log with timestamps and event labels.
    order_condition : str
        "O1" or "O2" for motor order.

    Returns
    -------
    pd.DataFrame
        Same samples as input with added:
        phase_id, motor_type, assembly_id, activity_id, label
    """

    # --- Basic validation ---------------------------------------------------
    if "abs_time" not in df.columns:
        raise ValueError("Accelerometer df is missing required column 'abs_time'.")
    if "timestamp" not in button_log.columns:
        raise ValueError("Button log is missing required column 'timestamp'.")
    if "event" not in button_log.columns:
        raise ValueError("Button log is missing 'event' column.")

    if order_condition not in ("O1", "O2"):
        raise ValueError("order_condition must be 'O1' or 'O2'.")

    # --- Prepare inputs -----------------------------------------------------
    labelled_df = df.copy()
    labelled_df["abs_time"] = pd.to_datetime(labelled_df["abs_time"], errors="coerce")

    events = button_log.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")

    # Apply the event parsing to all events in the log
    events[[
        "event_type",
        "motor_logged",
        "phase_id_logged",
        "activity_id_logged",
        "activity_label_logged"
    ]] = events["event"].apply(parse_event_text)

    # initialize new columns
    labelled_df[["label", "phase_id",
                 "motor_type", "assembly_id", "activity_id"]] = None

    # Motor sequence: depends on order condition
    def get_expected_motor(phase_idx): # Expected motor for given phase index
        if order_condition == "O1":
            return "M1" if (phase_idx % 2 == 1) else "M2"
        else:
            return "M2" if (phase_idx % 2 == 1) else "M1"

    # --- Extract event categories ---------------------------------------------
    phases = events[events["event_type"] == "Phase start"].sort_values("timestamp")
    assembly_starts = events[events["event_type"] == "Assembly start"].sort_values("timestamp")
    activity_ends = events[events["event_type"] == "Activity end"].sort_values("timestamp")
    assembly_ends = events[events["event_type"] == "Last Assembly Activity ended"].sort_values("timestamp")

    # Phase count check
    if len(phases) == 0:
        raise RuntimeError("No 'Phase start' events found.")
    else:
        print(f"{len(phases)} phases detected.")

    phases = phases.reset_index(drop=True)

    # --- Main labeling loop -------------------------------------------------
    for phase_index, phase_row in phases.iterrows():

        phase_id = phase_row["phase_id_logged"]
        motor_type = phase_row["motor_logged"]

        # Check for missing phases
        if phase_index+1 != phase_id:
            print(f"Warning: Missing phase {phase_index + 1}.")
            continue

        # Check motor type consistency
        expected_motor = get_expected_motor(phase_id)
        if motor_type != expected_motor:
            print(f"Warning: Unexpected motor at phase {phase_id}. Expected {expected_motor}, got {motor_type}.")

        # Timestamp for the start of this phase
        phase_start = phase_row["timestamp"]

        # phase end boundary
        next_phase = (
            phases.loc[phase_index + 1, "timestamp"]
            if phase_index + 1 < len(phases)
            else labelled_df["abs_time"].max()
        )

        # assemblies within this phase
        assemblies_in_phase = assembly_starts[
            (assembly_starts["timestamp"] > phase_start)
            & (assembly_starts["timestamp"] < next_phase)
        ]

        if assemblies_in_phase.empty:
            print(f"Warning: No assemblies found in phase {phase_id}.")
            continue
        
        # Baseline period: from phase start to first assembly start
        # ---------------------------------------------------------------
        first_assembly_time = assemblies_in_phase["timestamp"].iloc[0]
        mask = (labelled_df["abs_time"] >= phase_start) & (labelled_df["abs_time"] < first_assembly_time)

        labelled_df.loc[mask, ["label", "phase_id", "motor_type"]] = (
            "Baseline", phase_id, motor_type
        )

        # ---------------------------------------------------------------
        # Process individual assemblies
        # ---------------------------------------------------------------
        activities_for_motor = MOTOR1_ACTIVITIES if motor_type == "M1" else MOTOR2_ACTIVITIES

        for assembly_id, asm in enumerate(assemblies_in_phase.itertuples(), start=1):

            assembly_start = asm.timestamp
            assembly_end = assembly_ends[assembly_ends["timestamp"] > assembly_start]["timestamp"].min()

            # Determine the next assembly start (for boundary checks)
            next_asm_start = None # Stays at None if this is the last assembly in phase
            if assembly_id < len(assemblies_in_phase):
                next_asm_start = assemblies_in_phase.iloc[assembly_id].timestamp

            # ------ Boundary checks ------
            # Assembly did not complete at all
            if pd.isna(assembly_end):
                continue
            # Check for the case where assembly_end is before the next assembly start and therefore missing an explicit end
            elif next_asm_start and assembly_end >= next_asm_start:
                assembly_end = next_asm_start
                print(f"Warning: Assembly in phase {phase_id}, assembly {assembly_id} missing explicit end; using next assembly start as boundary.")
            # Check for the case where the last assembly in the phase is missing an explicit end and therefore exceeds the phase boundary
            elif assembly_end >= next_phase:
                assembly_end = next_phase
                print(f"Warning: Assembly in phase {phase_id}, assembly {assembly_id} exceeds phase boundary.")


            # Find activity events within this assembly
            asm_activity_events = activity_ends[
                (activity_ends["timestamp"] > assembly_start) &
                (activity_ends["timestamp"] <= assembly_end)
            ]

            # Build activity intervals:
            edges = [assembly_start] + asm_activity_events["timestamp"].tolist()
            ids = asm_activity_events["activity_id_logged"].tolist()
            labels = asm_activity_events["activity_label_logged"].tolist()

            # Report excess activities
            if len(edges) - 1 > len(activities_for_motor):
                print(f"Warning: More activity segments than labels for {motor_type} in phase {phase_id}, assembly {assembly_id}.")

            # Report missing expected activities
            logged_set = set(labels)
            expected_set = set(activities_for_motor)
            missing = expected_set - logged_set
            for m in missing:
                print(f"Missing activity in phase {phase_id}, assembly {assembly_id}: {m}")

            # label the actual segments
            for idx in range(len(edges) - 1):
                start = edges[idx]
                end = edges[idx + 1]

                act_id_logged = ids[idx]
                act_label_logged = labels[idx]

                mask = (labelled_df["abs_time"] >= start) & (labelled_df["abs_time"] < end)

                labelled_df.loc[mask, ["label",
                                       "phase_id",
                                       "motor_type",
                                       "assembly_id",
                                       "activity_id"]] = [
                    act_label_logged,
                    phase_id,
                    motor_type,
                    assembly_id,
                    act_id_logged
                ]

    # Remove rows with empty or unknown activity assignments
    labelled_df = labelled_df[
        labelled_df["label"].notna()
        & (labelled_df["label"] != "")
        & (labelled_df["label"] != "Unknown step")
    ]

    return labelled_df

def encode_labels(df: pd.DataFrame, results_folder: str | Path | None = None, str_label_column: str = "label") -> pd.DataFrame:
    # Create a global LabelEncoder on the pooled string labels
    # so encoding is consistent across all sessions.
    if str_label_column not in df.columns:
        raise ValueError(
            "DataFrame must contain a string label column (default 'label') to create label encoding."
        )
    unique_labels = sorted(df[str_label_column].dropna().unique())
    le = LabelEncoder()
    le.fit(unique_labels)
    df["label_encoded"] = le.transform(df[str_label_column])
    # Save the encoder centrally
    if results_folder is not None:
        le_path = Path(results_folder) / "label_encoder.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        print(f"Saved LabelEncoder to {le_path.resolve()}")
    return df

# ---------------------------------------------------------------------------
# Summary table generator
# ---------------------------------------------------------------------------
def build_label_summary(labelled_df: pd.DataFrame) -> dict:
    """
    Build detailed summary statistics from labeled accelerometer samples.

    This function generates multiple summary tables describing the duration,
    structure, and distribution of phases, assemblies, and activities in the
    labeled dataset.

    Output Overview
    ---------------
    1) activity_segments (per activity instance)
        One row per continuous activity segment defined by:
            (phase_id, assembly_id, activity_id, motor_type, label)
        Contains:
            - num_samples          Number of labeled samples in the segment
            - start_time           Earliest timestamp of the activity
            - end_time             Latest timestamp of the activity
            - duration             Duration in seconds

    2) phase_segments (per phase)
        One row per (phase_id, motor_type)
        Contains:
            - num_samples
            - start_time
            - end_time
            - duration

    3) assembly_segments (per assembly)
        One row per (phase_id, assembly_id, motor_type)
        Contains:
            - num_samples
            - start_time
            - end_time
            - duration

    4) phase_stats (aggregated per motor)
        Contains one row per motor_type:
            - number_of_phases
            - total_phase_duration
            - avg_phase_duration
            - std_phase_duration
            
    5) assembly_stats (aggregated per motor)
        Contains:
            - number_of_assemblies
            - total_assembly_duration
            - avg_assembly_duration
            - std_assembly_duration
            
    6) activity_stats (aggregated per activity)
        One row per (motor_type, activity_id, label):
            - avg_activity_duration_sec
            - total_activity_duration_sec
            - std_activity_duration_sec
            - n_segments

    7) activity_totals (global per activity)
        One row per (activity_id, label) describing total time spent
        on each activity across the entire dataset and the number of repetitions:
            - total_duration_sec
            - repetitions
            - avg_duration_sec
            - std_duration_sec

        This is useful for estimating the expected number of
        time windows per activity during HAR window segmentation.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Labeled dataset containing at least:
        - abs_time (datetime)
        - label
        - phase_id
        - assembly_id
        - activity_id
        - motor_type

    Returns
    -------
    dict
        {
            "activity_segments": DataFrame,
            "phase_segments": DataFrame,
            "assembly_segments": DataFrame,
            "phase_stats": DataFrame,
            "assembly_stats": DataFrame,
            "activity_stats": DataFrame,
            "activity_totals": DataFrame
        }
    """

    # Remove rows without an activity label
    labelled = labelled_df.dropna(subset=["label"])

    # ---------------------------------------------------
    # ACTIVITY SEGMENTS
    # ---------------------------------------------------
    activity_segments = (
        labelled
        .groupby(["phase_id", "assembly_id", "activity_id",
                  "motor_type", "label"])
        .agg(
            num_samples=("label", "size"),
            start_time=("abs_time", "min"),
            end_time=("abs_time", "max"),
        )
        .reset_index()
        .sort_values(["phase_id", "assembly_id", "activity_id"])
    )

    activity_segments["duration"] = (
        (activity_segments["end_time"] - activity_segments["start_time"])
        .dt.total_seconds()
    )

    # ---------------------------------------------------
    # PHASE SEGMENTS 
    # ---------------------------------------------------
    phase_segments = (
        labelled
        .groupby(["phase_id", "motor_type"])
        .agg(
            num_samples=("label", "size"),
            start_time=("abs_time", "min"),
            end_time=("abs_time", "max"),
        )
        .reset_index()
    )

    phase_segments["duration"] = (
        (phase_segments["end_time"] - phase_segments["start_time"])
        .dt.total_seconds()
    )

    # ---------------------------------------------------
    # ASSEMBLY SEGMENTS
    # ---------------------------------------------------
    assembly_segments = (
        labelled
        .groupby(["phase_id", "assembly_id", "motor_type"])
        .agg(
            num_samples=("label", "size"),
            start_time=("abs_time", "min"),
            end_time=("abs_time", "max"),
        )
        .reset_index()
    )

    assembly_segments["duration"] = (
        (assembly_segments["end_time"] - assembly_segments["start_time"])
        .dt.total_seconds()
    )

    # ---------------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------------

    # Average AND total phase duration per motor
    phase_stats = (
        phase_segments
        .groupby("motor_type")
        .agg(
            number_of_phases=("duration", "size"),
            total_phase_duration=("duration", "sum"),
            avg_phase_duration=("duration", "mean"),
            std_phase_duration=("duration", "std"),
        )
        .reset_index()
    )

    # Average AND total assembly duration per motor
    assembly_stats = (
        assembly_segments
        .groupby("motor_type")
        .agg(
            number_of_assemblies=("duration", "size"),
            total_assembly_duration=("duration", "sum"),
            avg_assembly_duration=("duration", "mean"),
            std_assembly_duration=("duration", "std"),
        )
        .reset_index()
    )

    # Average AND total activity duration per motor & activity
    activity_stats = (
        activity_segments
        .groupby(["motor_type", "activity_id", "label"])
        .agg(
            avg_activity_duration_sec=("duration", "mean"),
            total_activity_duration_sec=("duration", "sum"),
            std_activity_duration_sec=("duration", "std"),
            repetitions=("duration", "size"),
        )
        .reset_index()
        .sort_values(["motor_type", "activity_id"])
    )

    # ---------------------------------------------------
    # TOTAL ACTIVITY DURATION (for later window estimation)
    # ---------------------------------------------------
    activity_totals = (
        activity_segments
        .groupby(["activity_id", "label"])
        .agg(
            total_duration_sec=("duration", "sum"),
            repetitions=("duration", "size"),
            std_duration_sec=("duration", "std")
        )
        .reset_index()
        .sort_values("activity_id")
    )

    return {
        "activity_segments": activity_segments,
        "phase_segments": phase_segments,
        "assembly_segments": assembly_segments,
        "phase_stats": phase_stats,
        "assembly_stats": assembly_stats,
        "activity_stats": activity_stats,
        "activity_totals": activity_totals
    }
