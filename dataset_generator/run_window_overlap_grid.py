from __future__ import annotations

from typing import Iterable

# Add parent directory to path so absolute imports work
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_generator.core import PreprocessingConfig
from dataset_generator.run_dataset_generator import main as run_generator


# ============================================================
# Grid Configuration (edit these constants)
# ============================================================
WINDOW_RANGE_START_SEC = 3
WINDOW_RANGE_END_SEC = 3
OVERLAP_PERCENTAGES = ([75])


def validate_window_range(start: int, end: int) -> tuple[int, int]:
    """Validate inclusive integer window range."""
    if start <= 0 or end <= 0:
        raise ValueError("Window range bounds must be positive integers.")
    if end < start:
        raise ValueError("Window range end must be greater than or equal to start.")

    return start, end


def validate_overlap_percentages(overlaps: Iterable[float]) -> list[float]:
    """Validate overlap percentages from a numeric iterable."""
    values: list[float] = []
    for raw_pct in overlaps:
        pct = float(raw_pct)

        if pct < 0 or pct >= 100:
            raise ValueError(
                f"Overlap percentage must satisfy 0 <= overlap < 100. Got {pct}."
            )
        values.append(pct)

    if not values:
        raise ValueError("At least one overlap percentage is required.")

    # Keep insertion order while removing duplicates
    seen: set[float] = set()
    unique_values: list[float] = []
    for pct in values:
        if pct not in seen:
            seen.add(pct)
            unique_values.append(pct)

    return unique_values


def iter_window_sizes(start: int, end: int) -> Iterable[int]:
    """Yield inclusive integer window sizes in seconds."""
    for value in range(start, end + 1):
        yield value


def format_overlap_name(pct: float) -> str:
    """Format overlap value for user-facing logs."""
    if pct == int(pct):
        return str(int(pct))
    return str(pct)


def run_grid(window_range_start: int, window_range_end: int, overlaps: Iterable[float]) -> None:
    """Run preprocessing for every (window_size, overlap) combination."""
    start, end = validate_window_range(window_range_start, window_range_end)
    overlap_values = validate_overlap_percentages(overlaps)

    combinations: list[tuple[int, float, float]] = []
    for window_size_sec in iter_window_sizes(start, end):
        for overlap_pct in overlap_values:
            step_size_sec = window_size_sec * (1.0 - overlap_pct / 100.0)
            combinations.append((window_size_sec, overlap_pct, step_size_sec))

    print("=" * 80)
    print("WINDOW/OVERLAP GRID RUN")
    print("=" * 80)
    print(f"Window range: {start}-{end} seconds")
    print("Overlap percentages: " + ", ".join(format_overlap_name(v) for v in overlap_values))
    print(f"Total runs: {len(combinations)}")

    failures: list[tuple[int, float, str]] = []

    for idx, (window_size_sec, overlap_pct, step_size_sec) in enumerate(combinations, start=1):
        print("\n" + "-" * 80)
        print(
            f"Run {idx}/{len(combinations)}: "
            f"window={window_size_sec}s, overlap={format_overlap_name(overlap_pct)}%, step={step_size_sec}s"
        )
        print("-" * 80)

        config = PreprocessingConfig(
            full_activity_windowing=False,
            window_size_sec=float(window_size_sec),
            step_size_sec=float(step_size_sec),
        )

        try:
            run_generator(config)
        except Exception as exc:
            failures.append((window_size_sec, overlap_pct, str(exc)))
            print(
                f"Run failed for window={window_size_sec}s, "
                f"overlap={format_overlap_name(overlap_pct)}%: {exc}"
            )

    print("\n" + "=" * 80)
    print("GRID RUN COMPLETE")
    print("=" * 80)
    print(f"Total runs: {len(combinations)}")
    print(f"Successful: {len(combinations) - len(failures)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailed combinations:")
        for window_size_sec, overlap_pct, error_msg in failures:
            print(
                f"  - window={window_size_sec}s, overlap={format_overlap_name(overlap_pct)}% -> {error_msg}"
            )


def main() -> None:
    run_grid(
        window_range_start=WINDOW_RANGE_START_SEC,
        window_range_end=WINDOW_RANGE_END_SEC,
        overlaps=OVERLAP_PERCENTAGES,
    )


if __name__ == "__main__":
    main()
