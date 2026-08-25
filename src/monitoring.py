"""Population Stability Index (PSI) based drift detection for production monitoring.

PSI compares the distribution of a feature (or prediction) seen in production
against the distribution observed at training time. As a rule of thumb:

    PSI < 0.1  -> no significant drift
    0.1 <= PSI < 0.25 -> moderate drift, worth investigating
    PSI >= 0.25 -> significant drift, retraining is recommended

``DRIFT_THRESHOLD`` below is the value used to decide whether an alert should fire.
"""

import numpy as np

DRIFT_THRESHOLD = 0.25
NUM_BINS = 10


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = NUM_BINS) -> float:
    """Compute PSI between a reference (training) sample and a current (production) sample."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("Both reference and current samples must be non-empty")

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(reference.size, 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(current.size, 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_drift(reference: np.ndarray, current: np.ndarray, threshold: float = DRIFT_THRESHOLD) -> dict:
    """Return the PSI score plus a boolean flag for whether it exceeds ``threshold``."""
    psi = population_stability_index(reference, current)
    return {"psi": psi, "drift_detected": psi >= threshold, "threshold": threshold}
