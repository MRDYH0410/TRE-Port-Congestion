"""Frozen-input verification and mutually exclusive experiment path construction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_5_2_1 = CODE_ROOT / "experiments" / "5.2-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(namespace: str, index: int) -> int:
    payload = f"{namespace}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _load_interface_module():
    module_path = EXPERIMENT_5_2_1 / "interface.py"
    spec = importlib.util.spec_from_file_location("tre84_experiment_5_2_1_interface", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the unique 5.2.1 interface")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Frozen521Inputs:
    output_dir: Path
    historical: pd.DataFrame
    residuals: pd.DataFrame
    interface_hash: str
    residual_hash: str
    run_manifest_hash: str


@dataclass(frozen=True)
class PhysicalPath:
    path_id: str
    split: str
    frame: pd.DataFrame
    path_hash: str
    construction: str
    residual_start: str
    residual_end: str
    onset_week: int
    active_duration_weeks: int
    severity_floor: float
    has_reclosure: bool

    def manifest_record(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "split": self.split,
            "path_content_sha256": self.path_hash,
            "weeks": len(self.frame),
            "construction": self.construction,
            "residual_start": self.residual_start,
            "residual_end": self.residual_end,
            "onset_week": self.onset_week,
            "active_duration_weeks": self.active_duration_weeks,
            "severity_floor": self.severity_floor,
            "has_reclosure": self.has_reclosure,
            "historical_test_event_used": self.split == "test",
            "released_information_source": (
                "5.2.1 released_hmm_filter" if self.split == "test" else "designed split-specific training information"
            ),
        }


def load_frozen_5_2_1_inputs(config: Mapping[str, Any]) -> Frozen521Inputs:
    output_dir = (CODE_ROOT / str(config["input_5_2_1"])).resolve()
    acceptance_path = output_dir / "acceptance_5_2_1.json"
    manifest_path = output_dir / "run_manifest.json"
    with acceptance_path.open(encoding="utf-8") as stream:
        acceptance = json.load(stream)
    if acceptance.get("status") != "complete" or acceptance.get("blocking_failures"):
        raise RuntimeError("5.2.1 must be complete with no blocking failures before 5.2.2")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    outputs = {item["path"]: item for item in manifest["outputs"] if "sha256" in item}
    residual_name = "counterfactual_residual_library.csv"
    interface_name = "historical_information_event_path.csv"
    for required in (residual_name, interface_name):
        if required not in outputs:
            raise RuntimeError(f"5.2.1 manifest does not freeze {required}")
    expected_interface_hash = str(config["expected_5_2_1_interface_sha256"])
    actual_interface_hash = sha256_file(output_dir / interface_name)
    if actual_interface_hash != expected_interface_hash:
        raise RuntimeError(
            "The unique 5.2.1 historical interface hash does not match the "
            "authorised SHA256"
        )
    if outputs[interface_name]["sha256"] != expected_interface_hash:
        raise RuntimeError("The 5.2.1 manifest and authorised interface hash disagree")
        if sha256_file(output_dir / required) != outputs[required]["sha256"]:
            raise RuntimeError(f"5.2.1 frozen output hash mismatch: {required}")
    interface = _load_interface_module()
    historical = interface.load_historical_path(output_dir)
    residuals = pd.read_csv(
        output_dir / residual_name,
        parse_dates=["residual_date", "origin_date", "training_cutoff"],
    )
    event_start = pd.Timestamp(historical["week"].min())
    if residuals["residual_date"].max() >= event_start:
        raise RuntimeError("The test bootstrap library contains event-period residuals")
    if residuals["forecast_horizon"].ne(1).any() or residuals["residual_date"].duplicated().any():
        raise RuntimeError("5.2.2 requires the unique pre-event one-step residual library")
    return Frozen521Inputs(
        output_dir=output_dir,
        historical=historical,
        residuals=residuals,
        interface_hash=outputs[interface_name]["sha256"],
        residual_hash=outputs[residual_name]["sha256"],
        run_manifest_hash=sha256_file(manifest_path),
    )


def _canonical_path_hash(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    canonical = frame.loc[:, columns].copy()
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].dt.strftime("%Y-%m-%d")
    text = canonical.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _residual_block(residuals: pd.DataFrame, seed: int, horizon: int) -> pd.DataFrame:
    ordered = residuals.sort_values("residual_date").reset_index(drop=True)
    if len(ordered) < horizon:
        raise RuntimeError("The pre-event residual library is too short")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(ordered) - horizon + 1))
    return ordered.iloc[start : start + horizon].reset_index(drop=True)


def _synthetic_information(
    config: Mapping[str, Any],
    split: str,
    index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    design = config["paths"]["designed_information"]
    anchors = tuple(float(value) for value in design[f"{split}_current_risk_anchors"])
    initial = float(anchors[index % len(anchors)])
    current = np.full(horizon, initial, dtype=float)
    lead = np.clip(
        float(design["lead_forecast_current_weight"]) * current
        + float(design["lead_forecast_intercept"]),
        0.0,
        1.0,
    )
    return current, lead


def _designed_serviceability(
    config: Mapping[str, Any],
    split: str,
    index: int,
    horizon: int,
) -> tuple[np.ndarray, int, int, float, bool]:
    design = config["paths"]["designed_serviceability"]
    onsets = tuple(int(value) for value in design[f"{split}_onset_weeks"])
    floors = tuple(float(value) for value in design[f"{split}_severity_floors"])
    durations = tuple(int(value) for value in design[f"{split}_duration_weeks"])
    reclosure_indices = {
        int(value) for value in design[f"{split}_reclosure_path_indices"]
    }
    onset = onsets[index % len(onsets)]
    reclosure = index in reclosure_indices
    floor = float(floors[index % len(floors)])
    duration = int(durations[index % len(durations)])
    values = np.ones(horizon, dtype=float)
    active_end = min(onset + duration, horizon)
    values[onset:active_end] = floor
    remaining = horizon - active_end
    if remaining > 0:
        values[active_end:] = floor + (1.0 - floor) * (
            np.arange(1, remaining + 1, dtype=float) / remaining
        )
    if reclosure:
        values[-int(design["reclosure_tail_weeks"]) :] = floor
    return values, onset, duration, floor, reclosure


def build_training_validation_paths(
    *,
    config: Mapping[str, Any],
    residuals: pd.DataFrame,
    reference_normal_model_units: float,
) -> tuple[list[PhysicalPath], list[PhysicalPath]]:
    horizon = int(config["event_weeks"])
    exposure_frame = pd.read_csv(
        CODE_ROOT / "experiments" / "data" / "processed" / "anchors" / "network_exposure_reference.csv"
    )
    exposure = float(exposure_frame.loc[0, "reference_network_exposure"])
    unit = float(config["model_unit_tonnes"])
    results: dict[str, list[PhysicalPath]] = {"training": [], "validation": []}
    calendar = config["paths"]["synthetic_calendar"]
    base_dates = {
        "training": pd.Timestamp(calendar["training_start_monday"]),
        "validation": pd.Timestamp(calendar["validation_start_monday"]),
    }
    for split, count_key, namespace_key in (
        ("training", "training_count", "training_seed_namespace"),
        ("validation", "validation_count", "validation_seed_namespace"),
    ):
        count = int(config["paths"][count_key])
        for index in range(count):
            block = _residual_block(
                residuals,
                deterministic_seed(str(config["paths"][namespace_key]), index),
                horizon,
            )
            normal = np.maximum(
                reference_normal_model_units + block["residual"].to_numpy(dtype=float) * exposure / unit,
                1e-6,
            )
            service, onset, duration, floor, reclosure = _designed_serviceability(
                config, split, index, horizon
            )
            current_risk, lead_risk = _synthetic_information(
                config, split, index, horizon
            )
            weeks = pd.date_range(
                base_dates[split]
                + pd.Timedelta(
                    weeks=int(calendar["path_spacing_weeks"]) * index
                ),
                periods=horizon,
                freq="W-MON",
            )
            frame = pd.DataFrame(
                {
                    "week": weeks,
                    "normal_model_units": normal,
                    "serviceability": service,
                    "filtered_high_risk_probability": current_risk,
                    "lead_time_high_risk_probability": lead_risk,
                    "release_date": weeks,
                    "source_observation_month": weeks.to_period("M").to_timestamp(),
                    "timing_valid": True,
                    "information_source": "designed_training_information",
                    "residual_date": block["residual_date"].to_numpy(),
                }
            )
            path_hash = _canonical_path_hash(frame)
            results[split].append(
                PhysicalPath(
                    path_id=f"{split}_{index:03d}_{path_hash[:12]}",
                    split=split,
                    frame=frame,
                    path_hash=path_hash,
                    construction="designed event geometry plus 5.2.1 pre-event residual block",
                    residual_start=str(block["residual_date"].min().date()),
                    residual_end=str(block["residual_date"].max().date()),
                    onset_week=onset,
                    active_duration_weeks=duration,
                    severity_floor=floor,
                    has_reclosure=reclosure,
                )
            )
    train_hashes = {path.path_hash for path in results["training"]}
    validation_hashes = {path.path_hash for path in results["validation"]}
    if train_hashes & validation_hashes:
        raise RuntimeError("Training and validation physical paths overlap")
    return results["training"], results["validation"]


def build_test_paths(
    *,
    config: Mapping[str, Any],
    frozen: Frozen521Inputs,
    count: int,
) -> list[PhysicalPath]:
    historical = frozen.historical.sort_values("week").reset_index(drop=True)
    horizon = int(config["event_weeks"])
    if len(historical) != horizon:
        raise RuntimeError("The frozen historical interface must have 21 weeks")
    ordered_residuals = frozen.residuals.sort_values("residual_date").reset_index(drop=True)
    available_starts = len(ordered_residuals) - horizon + 1
    if count > available_starts:
        raise RuntimeError(
            f"Requested {count} unique test paths but only {available_starts} "
            "contiguous residual blocks are available"
        )
    # Preserve the pre-existing deterministic candidate for every index.  A
    # deterministic fallback is used only when that candidate repeats an
    # earlier start, thereby extending the design without replacing any valid
    # unique pilot path.
    selected_starts: list[int] = []
    used_starts: set[int] = set()
    namespace = str(config["paths"]["test_seed_namespace"])
    for index in range(count):
        seed = deterministic_seed(namespace, index)
        candidate = int(np.random.default_rng(seed).integers(0, available_starts))
        if candidate in used_starts:
            fallback_seed = deterministic_seed(f"{namespace}|duplicate-resolution", index)
            candidates = np.random.default_rng(fallback_seed).permutation(available_starts)
            candidate = next(int(value) for value in candidates if int(value) not in used_starts)
        selected_starts.append(candidate)
        used_starts.add(candidate)
    paths: list[PhysicalPath] = []
    for index, start in enumerate(selected_starts):
        block = ordered_residuals.iloc[start : start + horizon].reset_index(drop=True)
        counterfactual = np.maximum(
            historical["estimated_no_disruption_activity"].to_numpy(dtype=float)
            + block["residual"].to_numpy(dtype=float),
            historical["observed_activity"].to_numpy(dtype=float),
        )
        observed = historical["observed_activity"].to_numpy(dtype=float)
        serviceability = np.divide(
            observed,
            counterfactual,
            out=np.ones_like(observed),
            where=counterfactual > 0,
        )
        serviceability = np.clip(serviceability, 0.0, 1.0)
        normal_model_units = (
            historical["network_exposure_reference"].to_numpy(dtype=float)
            * counterfactual
            / historical["model_unit_tonnes"].to_numpy(dtype=float)
        )
        frame = pd.DataFrame(
            {
                "week": historical["week"],
                "normal_model_units": normal_model_units,
                "serviceability": serviceability,
                "filtered_high_risk_probability": historical["filtered_high_risk_probability"],
                "lead_time_high_risk_probability": historical["lead_time_high_risk_probability"],
                "release_date": historical["release_date"],
                "source_observation_month": historical["source_observation_month"],
                "timing_valid": historical["timing_valid"],
                "information_source": historical["risk_information_source"],
                "residual_date": block["residual_date"].to_numpy(),
            }
        )
        if (frame["release_date"] > frame["week"]).any() or not frame["timing_valid"].all():
            raise RuntimeError("A test path used information after its decision week")
        path_hash = _canonical_path_hash(frame)
        active = frame["serviceability"].to_numpy(dtype=float) < 1.0 - 1e-12
        onset = int(np.argmax(active)) if active.any() else horizon
        paths.append(
            PhysicalPath(
                path_id=f"test_{index:03d}_{path_hash[:12]}",
                split="test",
                frame=frame,
                path_hash=path_hash,
                construction="5.2.1 historical information path plus matched pre-event residual block",
                residual_start=str(block["residual_date"].min().date()),
                residual_end=str(block["residual_date"].max().date()),
                onset_week=onset,
                active_duration_weeks=int(active.sum()),
                severity_floor=float(frame["serviceability"].min()),
                has_reclosure=bool(any((~active[:-1]) & active[1:])) if len(active) > 1 else False,
            )
        )
    hashes = [path.path_hash for path in paths]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Test physical paths need unique content hashes")
    return paths


def manifest_frame(paths: Sequence[PhysicalPath]) -> pd.DataFrame:
    return pd.DataFrame([path.manifest_record() for path in paths])
