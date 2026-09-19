from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import logging

from .signal_preprocessing_utils import get_logger

def grouped_stratified_split_groups(
    groups: Dict[str, Dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    tolerance: float = 0.07,
    max_attempts: int = 1000,
    random_state: Optional[int] = 42,
    logger: Optional[logging.Logger] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Grouped stratified splitting by (phase, assembly) combos.
    """
    logger = get_logger(logger)

    rng = np.random.RandomState(random_state)

    # Map combos -> gids
    comb_to_gid = {}
    for gid, g in groups.items():
        combo = (g["session"], g["phase"], g["assembly"])
        comb_to_gid.setdefault(combo, []).append(gid)

    combs = list(comb_to_gid.keys())

    # Global label proportions
    labels = [g["label"] for g in groups.values()]
    ulabels, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    global_props = {u: c / total for u, c in zip(ulabels, counts)}

    # Deviation function to see how well the label proportion 
    # in a set of gids (group IDs, which are synonymous with the IDs of windows) matches global proportions
    # => See how much the proportions in a set of windows deviate from the overall proportions
    def prop_dev(gids):
        if len(gids) == 0:
            return 1.0
        # Extract all the labels present in the set of gids
        labs = [groups[g]["label"] for g in gids]
        u, c = np.unique(labs, return_counts=True)
        # Calculate local proportions
        local = {uu: cc / sum(c) for uu, cc in zip(u, c)}
        # Return the deviation of local from global proportions for each label
        return max(abs(local.get(x, 0) - global_props.get(x, 0)) for x in global_props)

    # Search for best split (loweest worst-case deviation from global proportions)
    best = None
    # Save the best score found so far
    best_score = float("inf")

    # Try multiple random splits until the tolerance is met or max attempts reached
    for _ in range(max_attempts):
        shuffled = combs.copy()
        rng.shuffle(shuffled)

        # Build train block by adding combos (unique phase/assembly combinations) until size reached
        acc = 0
        train_combs = []
        for c in shuffled:
            train_combs.append(c)
            acc += len(comb_to_gid[c])
            if acc >= len(groups) * train_frac:
                break

        # Build val/test from remaining combos
        remain = [c for c in shuffled if c not in train_combs]
        if not remain:
            continue

        # Split remaining into val/test according to their relative sizes
        val_ratio = val_frac / (val_frac + test_frac)
        n_val = max(1, int(round(val_ratio * len(remain))))

        val_combs = remain[:n_val]
        test_combs = remain[n_val:]

        # Get gids for each split
        train_gids = sum((comb_to_gid[c] for c in train_combs), [])
        val_gids = sum((comb_to_gid[c] for c in val_combs), [])
        test_gids = sum((comb_to_gid[c] for c in test_combs), [])

        # Evaluate worst-case deviation from global proportions across train/val/test splits
        worst = max(prop_dev(train_gids), prop_dev(val_gids), prop_dev(test_gids))

        # Update best split if improved
        if worst < best_score:
            best_score = worst
            best = (train_gids, val_gids, test_gids)
            if worst <= tolerance:
                break

    if best is None:
        raise RuntimeError("Failed to find suitable stratified split")

    train_g, val_g, test_g = best

    if verbose:
        logger.info(f"[grouped_stratified_split_groups] train={len(train_g)}, val={len(val_g)}, test={len(test_g)}, worst_dev={best_score:.4f}")

    return (
        {gid: groups[gid] for gid in train_g},
        {gid: groups[gid] for gid in val_g},
        {gid: groups[gid] for gid in test_g}
    )


