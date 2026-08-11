"""The only supported loader for the frozen 5.2.1 historical information path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = CODE_ROOT / "output" / "5.2.1_data_event_information_validity"
INTERFACE_FILENAME = "historical_information_event_path.csv"
REQUIRED_COLUMNS = (
    "week",
    "observed_activity",
    "estimated_no_disruption_activity",
    "serviceability",
    "blocked_activity_proxy",
    "model_blocked_units",
    "source_observation_month",
    "release_date",
    "filtered_high_risk_probability",
    "lead_time_high_risk_probability",
    "monthly_transitions_to_maturity",
    "timing_valid",
    "risk_information_source",
    "event_path_source",
    "contains_committed_share",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_historical_path(output_dir: Path | str = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load and hash-verify the frozen interface; never reconstruct a risk ramp."""

    output = Path(output_dir).resolve()
    path = output / INTERFACE_FILENAME
    manifest_path = output / "run_manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    records = {
        item["path"]: item for item in manifest["outputs"] if "sha256" in item
    }
    if INTERFACE_FILENAME not in records:
        raise ValueError("The run manifest does not freeze the historical interface")
    if _sha256(path) != records[INTERFACE_FILENAME]["sha256"]:
        raise ValueError("The historical information path does not match its run manifest")
    frame = pd.read_csv(
        path,
        parse_dates=[
            "week",
            "source_observation_month",
            "release_date",
            "decision_cutoff",
            "readiness_maturity_date",
        ],
    )
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Historical information interface is missing columns: {missing}")
    if len(frame) != 21 or frame["week"].duplicated().any():
        raise ValueError("The frozen historical path must contain 21 unique event weeks")
    if not frame["timing_valid"].all() or (frame["release_date"] > frame["week"]).any():
        raise ValueError("The historical interface contains unreleased information")
    if set(frame["risk_information_source"]) != {"released_hmm_filter"}:
        raise ValueError("Artificial risk paths are forbidden; use the released HMM filter")
    if set(frame["event_path_source"]) != {"formula_derived_5.2.1"}:
        raise ValueError("The historical event path must be the frozen 5.2.1 construction")
    if frame["contains_committed_share"].astype(bool).any():
        raise ValueError("Committed share is outside the 5.2.1 interface")
    return frame
