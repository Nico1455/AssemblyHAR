from __future__ import annotations

import pandas as pd

from .signal_preprocessing_utils import combine_activity_totals, concat_session_results


def compute_dataset_activity_statistics(
    all_activity_segments: list[pd.DataFrame],
    all_activity_totals: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute dataset-level activity segments and totals.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (global_activity_segments_df, global_activity_totals_df)
    """
    # Combine activity segments across all sessions
    global_activity_segments_df = concat_session_results(
        all_activity_segments,
        parse_dates=["start_time", "end_time"],
        compute_duration=False,
        required_cols=["label"],
    )

    # Combine activity totals across all sessions
    global_activity_totals_df = combine_activity_totals(
        all_activity_totals,
        activity_segments_df=global_activity_segments_df,
    )

    return global_activity_segments_df, global_activity_totals_df


def compute_dataset_phase_statistics(
    all_phase_stats: list[pd.DataFrame],
    global_activity_segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute dataset-level phase statistics per motor_type.
    """
    if not all_phase_stats:
        return pd.DataFrame()

    phase_df_all = concat_session_results(
        all_phase_stats,
        required_cols=["motor_type", "number_of_phases", "total_phase_duration"],
    )

    # Aggregate totals per motor from per-session summaries
    global_phase_stats_df = (
        phase_df_all.groupby("motor_type")
        .agg(
            total_phases=("number_of_phases", "sum"),
            total_phase_duration=("total_phase_duration", "sum"),
        )
        .reset_index()
    )

    if global_activity_segments_df is None or global_activity_segments_df.empty:
        global_phase_stats_df["std_phase_duration"] = pd.NA
    else:
        # Compute std across singular phase durations using activity segments
        # 1) Per phase duration per session: sum of activity segment durations
        #    for each (motor_type, phase_id)
        phase_duration_per_phase = (
            global_activity_segments_df
            .groupby(["motor_type", "session_code", "phase_id"])
            .agg(phase_duration=("duration", "sum"))
            .reset_index()
        )
        std_phase = (
            phase_duration_per_phase
            .groupby("motor_type")["phase_duration"].std()
            .reset_index()
            .rename(columns={"phase_duration": "std_phase_duration"})
        )
        global_phase_stats_df = global_phase_stats_df.merge(std_phase, on="motor_type", how="left")

    global_phase_stats_df["avg_phase_duration"] = (
        global_phase_stats_df["total_phase_duration"] / global_phase_stats_df["total_phases"]
    )
    return global_phase_stats_df


def compute_dataset_assembly_statistics(
    all_assembly_stats: list[pd.DataFrame],
    global_activity_segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute dataset-level assembly statistics per motor_type.
    """
    if not all_assembly_stats:
        return pd.DataFrame()

    assembly_df_all = concat_session_results(
        all_assembly_stats,
        required_cols=["motor_type", "number_of_assemblies", "total_assembly_duration"],
    )

    # Aggregate totals per motor from per-session summaries
    global_assembly_stats_df = (
        assembly_df_all.groupby("motor_type")
        .agg(
            total_assemblies=("number_of_assemblies", "sum"),
            total_assembly_duration=("total_assembly_duration", "sum"),
        )
        .reset_index()
    )

    if global_activity_segments_df is None or global_activity_segments_df.empty:
        global_assembly_stats_df["std_assembly_duration"] = pd.NA
    else:
        # Compute std across singular assembly durations using activity segments
        # Per-session assembly instances: (motor_type, session_code, phase_id, assembly_id)
        assembly_duration_per_assembly = (
            global_activity_segments_df
            .groupby(["motor_type", "session_code", "phase_id", "assembly_id"])
            .agg(assembly_duration=("duration", "sum"))
            .reset_index()
        )
        std_assembly = (
            assembly_duration_per_assembly
            .groupby("motor_type")["assembly_duration"].std()
            .reset_index()
            .rename(columns={"assembly_duration": "std_assembly_duration"})
        )
        global_assembly_stats_df = global_assembly_stats_df.merge(std_assembly, on="motor_type", how="left")

    global_assembly_stats_df["avg_assembly_duration"] = (
        global_assembly_stats_df["total_assembly_duration"] / global_assembly_stats_df["total_assemblies"]
    )
    return global_assembly_stats_df


def compute_phase_assembly_duration_breakdown(
    activity_segments_df: pd.DataFrame,
    phase_ids: tuple[int, ...] = (1, 2, 3, 4),
    phase1_assembly_ids: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """
    Compute per-motor duration statistics for the initial phases and assemblies.

    Metrics computed (mean/std):
      - phase durations for each phase_id in ``phase_ids``
      - first/second construction durations per session and motor_type
      - assembly durations for phase_id == 1 and assembly_id in ``phase1_assembly_ids``

    The computation uses two levels of per-instance durations:
    - phase instances grouped by (motor_type, session_code, phase_id)
    - assembly instances grouped by (motor_type, session_code, phase_id, assembly_id)
    and then aggregates each metric by motor_type.

    First/second construction logic:
    The idea here is to look at the difference between the first and second time a motor is built in a session.
    - For each (motor_type, session_code), phase instances are ordered by phase_id.
    - The earliest phase in that order contributes to first_construction_duration.
    - The second phase in that order contributes to second_construction_duration.
    This makes the metric robust to O1/O2: whichever phase index corresponds to
    the first build of a motor in that session is selected automatically.
    """
    if activity_segments_df is None or activity_segments_df.empty:
        return pd.DataFrame()

    df = activity_segments_df.copy()

    # Build grouping keys
    base_keys = ["motor_type", "session_code"]

    # Build keys for phase-instances
    phase_group_keys = base_keys + ["phase_id"]
    # Compute duration per phase instance (grouped by motor/session/phase)
    phase_instance_df = (
        df.groupby(phase_group_keys, dropna=False)
        .agg(duration=("duration", "sum"))
        .reset_index()
    )

    # Assembly-instance duration (used for phase 1 assembly stats)
    assembly_group_keys = base_keys + ["phase_id", "assembly_id"]
    # Compute duration per assembly instance (grouped by motor/session/phase/assembly)
    assembly_instance_df = (
        df.groupby(assembly_group_keys, dropna=False)
        .agg(duration=("duration", "sum"))
        .reset_index()
    )

    # Aggregate metrics by motor_type
    rows: list[dict] = []
    # Iterate over each motor_type group in the phase_instance_df
    for motor, g in phase_instance_df.groupby("motor_type"):
        row: dict[str, float | str] = {"motor_type": motor}
        g_assembly = assembly_instance_df[assembly_instance_df["motor_type"] == motor]

        # Phase 1..4 avg/std durations
        for phase_id in phase_ids:
            vals = g.loc[g["phase_id"] == phase_id, "duration"]
            row[f"avg_phase_{phase_id}_duration"] = vals.mean() if not vals.empty else pd.NA
            row[f"std_phase_{phase_id}_duration"] = vals.std() if len(vals) > 1 else pd.NA

        # Durations of first and second construction phases of a motor per session
        first_construction_vals: list[float] = []
        second_construction_vals: list[float] = []

        # Determine first/second construction phase durations per session.
        # If more than two constructions exist, use only the first two.
        for _, session_group in g.groupby("session_code"):
            ordered = session_group.sort_values("phase_id")
            if ordered.empty:
                continue
            first_construction_vals.append(float(ordered.iloc[0]["duration"]))
            if len(ordered) > 1:
                second_construction_vals.append(float(ordered.iloc[1]["duration"]))
       
        # Convert to Series for mean/std computation, handling empty cases
        first_series = pd.Series(first_construction_vals)
        second_series = pd.Series(second_construction_vals)

        row["avg_first_construction_duration"] = (
            first_series.mean() if not first_series.empty else pd.NA
        )
        row["std_first_construction_duration"] = (
            first_series.std() if len(first_series) > 1 else pd.NA
        )
        row["avg_second_construction_duration"] = (
            second_series.mean() if not second_series.empty else pd.NA
        )
        row["std_second_construction_duration"] = (
            second_series.std() if len(second_series) > 1 else pd.NA
        )

        # Assembly 1..3 of phase 1 avg/std durations
        for assembly_id in phase1_assembly_ids:
            vals = g_assembly.loc[
                (g_assembly["phase_id"] == 1) & (g_assembly["assembly_id"] == assembly_id),
                "duration",
            ]
            row[f"avg_phase1_assembly_{assembly_id}_duration"] = vals.mean() if not vals.empty else pd.NA
            row[f"std_phase1_assembly_{assembly_id}_duration"] = vals.std() if len(vals) > 1 else pd.NA

        rows.append(row)

    return pd.DataFrame(rows)


def compute_dataset_acceleration_statistics(session_acceleration_stats: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate accelerometer statistics across all sessions from session acceleration stats.

    Parameters
    ----------
    session_acceleration_stats : list[pd.DataFrame]
        Per-session tables (one row per axis and motor) with columns:
        motor_type, axis, count, min, max, mean, std, median, q25, q75, iqr, p5, p95.

    Returns
    -------
    pd.DataFrame
        One row per (motor_type, axis) with dataset-level aggregates.
    """
    if not session_acceleration_stats:
        return pd.DataFrame()

    all_stats = pd.concat(session_acceleration_stats, ignore_index=True)
    required = {
        "motor_type", "axis", "count", "min", "max", "mean", "std",
        "median", "q25", "q75", "iqr", "p5", "p95", "session_code"
    }
    if all_stats.empty or not required.issubset(all_stats.columns):
        return pd.DataFrame()

    per_axis = (
        all_stats.groupby(["motor_type", "axis"], dropna=False)
        .agg(
            n_sessions=("session_code", "nunique"),
            total_samples=("count", "sum"),
            weighted_mean_g=("mean", lambda s: (s * all_stats.loc[s.index, "count"]).sum() / all_stats.loc[s.index, "count"].sum()),
            mean_of_session_means_g=("mean", "mean"),
            std_of_session_means_g=("mean", "std"),
            global_min_g=("min", "min"),
            global_max_g=("max", "max"),
            pooled_std_g=("std", lambda s: (
                (
                    all_stats.loc[s.index, "count"] * (
                        s ** 2
                        + (
                            all_stats.loc[s.index, "mean"]
                            - (
                                all_stats.loc[s.index, "mean"] * all_stats.loc[s.index, "count"]
                            ).sum()
                            / all_stats.loc[s.index, "count"].sum()
                        ) ** 2
                    )
                ).sum()
                / all_stats.loc[s.index, "count"].sum()
            ) ** 0.5),
            mean_session_median_g=("median", "mean"),
            mean_session_iqr_g=("iqr", "mean"),
            mean_session_p5_g=("p5", "mean"),
            mean_session_p95_g=("p95", "mean"),
        )
        .reset_index()
    )

    return per_axis

def compute_dataset_acceleration_statistics(session_acceleration_stats: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate accelerometer statistics across all sessions from session acceleration stats.

    Parameters
    ----------
    session_acceleration_stats : list[pd.DataFrame]
        Per-session tables (one row per axis and motor) with columns:
        motor_type, axis, count, min, max, mean, std, median, q25, q75, iqr, p5, p95.

    Returns
    -------
    pd.DataFrame
        One row per (motor_type, axis) with dataset-level aggregates.
    """
    if not session_acceleration_stats:
        return pd.DataFrame()

    all_stats = pd.concat(session_acceleration_stats, ignore_index=True)
    required = {
        "motor_type", "axis", "count", "min", "max", "mean", "std",
        "median", "q25", "q75", "iqr", "p5", "p95", "session_code"
    }
    if all_stats.empty or not required.issubset(all_stats.columns):
        return pd.DataFrame()

    per_axis = (
        all_stats.groupby(["motor_type", "axis"], dropna=False)
        .agg(
            n_sessions=("session_code", "nunique"),
            total_samples=("count", "sum"),
            
            # The weighted mean is computed by multiplying each session's mean by its count, summing these, and dividing by the total count across sessions
            # This gives the exact global mean across all samples, not just the mean of session means
            weighted_mean_g=("mean", lambda s: (s * all_stats.loc[s.index, "count"]).sum() / all_stats.loc[s.index, "count"].sum()),
            
            mean_of_session_means_g=("mean", "mean"),
            std_of_session_means_g=("mean", "std"),
            global_min_g=("min", "min"),
            global_max_g=("max", "max"),
            
            # Pooled std is computed using the formula for pooled standard deviation across sessions
            # The pooled standard deviation is the exact global standard deviation across all samples
            pooled_std_g=("std", lambda s: (
                (
                    all_stats.loc[s.index, "count"] * (
                        s ** 2
                        + (
                            all_stats.loc[s.index, "mean"]
                            - (
                                all_stats.loc[s.index, "mean"] * all_stats.loc[s.index, "count"]
                            ).sum()
                            / all_stats.loc[s.index, "count"].sum()
                        ) ** 2
                    )
                ).sum()
                / all_stats.loc[s.index, "count"].sum()
            ) ** 0.5),
            
            mean_session_median_g=("median", "mean"),
            mean_session_iqr_g=("iqr", "mean"),
            mean_session_p5_g=("p5", "mean"),
            mean_session_p95_g=("p95", "mean"),
        )
        .reset_index()
    )

    return per_axis