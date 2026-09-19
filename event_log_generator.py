from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import Event, EventLog, Trace

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_SESSIONS_ROOT = PROJECT_ROOT / "Processed_Data" / "Processed_Session_Data"
EVENT_LOGS_DIR = PROJECT_ROOT / "Event_Logs"


def create_session_event_log(session_name: str, activity_segments_df: pd.DataFrame) -> EventLog:
    """Create one PM4Py EventLog for a single session from its activity_segments CSV."""
    df = activity_segments_df.copy()
    if df.empty:
        return EventLog()

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    if "activity_id" in df.columns:
        df = df.sort_values(["phase_id", "assembly_id", "activity_id"], kind="mergesort")
    else:
        df = df.sort_values(["phase_id", "assembly_id", "start_time"], kind="mergesort")

    event_log = EventLog()

    for (phase_id, assembly_id), group in df.groupby(["phase_id", "assembly_id"], sort=True):
        trace = Trace()
        case_id = f"{session_name}_Ph{int(float(phase_id))}_As{int(float(assembly_id))}"
        trace.attributes["concept:name"] = case_id
        trace.attributes["session_name"] = session_name
        trace.attributes["phase_id"] = int(float(phase_id))
        trace.attributes["assembly_id"] = int(float(assembly_id))

        for _, row in group.iterrows():
            event = Event(
                {
                    "concept:name": str(row["label"]),
                    "time:timestamp": row["end_time"].to_pydatetime(),
                    "phase_id": int(float(row["phase_id"])),
                    "assembly_id": int(float(row["assembly_id"])),
                    "motor_type": str(row.get("motor_type", "")),
                    "activity_id": row.get("activity_id"),
                    "num_samples": int(row.get("num_samples", 0)),
                    "start_time": pd.Timestamp(row["start_time"]).isoformat(),
                    "end_time": pd.Timestamp(row["end_time"]).isoformat(),
                    "duration": float(row.get("duration", 0.0)),
                }
            )
            trace.append(event)

        event_log.append(trace)

    return event_log


def generate_event_logs() -> list[dict[str, object]]:
    EVENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for session_dir in sorted(PROCESSED_SESSIONS_ROOT.glob("session_*_data")):
        segment_file = next(session_dir.joinpath("results").glob("*activity_segments.csv"), None)
        if segment_file is None:
            continue

        df = pd.read_csv(segment_file)
        if df.empty:
            continue

        session_name = session_dir.name
        event_log = create_session_event_log(session_name, df)
        if len(event_log) == 0:
            continue

        output_path = session_dir / f"{session_name}_event_log.xes"
        xes_exporter.apply(event_log, str(output_path))

        generated.append(
            {
                "session_name": session_name,
                "num_traces": len(event_log),
                "num_events": sum(len(trace) for trace in event_log),
            }
        )

    summary_path = PROCESSED_SESSIONS_ROOT / "dataset_event_log_summary.json"
    summary_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    return generated


def main() -> None:
    result = generate_event_logs()
    print(f"Generated {len(result)} session event logs in {PROCESSED_SESSIONS_ROOT}")
    for item in result:
        print(f"- {item['session_name']}")


if __name__ == "__main__":
    main()
