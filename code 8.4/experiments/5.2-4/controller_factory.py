"""Symmetric controller construction and capacity-right restrictions."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tre84.actions import Action

from features import LinearActor, state_feature_names
from information_design import InformationProvider, load_hmm_inputs
from model import BenchmarkModel, build_model
from paths import build_training_validation_paths, load_frozen_5_2_1_inputs, sha256_file
from policies import ModelGuidedPolicy
from training import generate_teacher_data, save_checkpoint, train_bc, train_sac


RIGHTS = {
    "RD": (True, True),
    "R": (True, False),
    "D": (False, True),
    "NONE": (False, False),
}


def controller_id(information_regime: str, capacity_rights: str) -> str:
    return f"{information_regime}_{capacity_rights}"


class CapacityRightsProjector:
    """Apply rights after a raw proposal and before the common projector."""

    def __init__(
        self,
        *,
        inner: Any,
        model: BenchmarkModel,
        capacity_rights: str,
    ) -> None:
        if capacity_rights not in RIGHTS:
            raise ValueError(f"Unknown capacity-right set {capacity_rights}")
        self.inner = inner
        self.model = model
        self.capacity_rights = capacity_rights
        self.readiness_allowed, self.direct_allowed = RIGHTS[capacity_rights]

    def restrict(self, raw_action: Action) -> Action:
        values = raw_action.vector(self.model.layout.keys).copy()
        if not self.readiness_allowed:
            for key in (*self.model.layout.readiness_order, *self.model.layout.readiness_exercise):
                values[self.model.layout.keys.index(key)] = 0.0
        if not self.direct_allowed:
            for key in self.model.layout.direct_order:
                values[self.model.layout.keys.index(key)] = 0.0
        return Action.from_vector(self.model.layout.keys, values)

    def project(self, raw_action: Action, state: Any) -> Any:
        return self.inner.project(self.restrict(raw_action), state)

    def local_jacobian(
        self,
        raw_action: Action,
        state: Any,
        *,
        projection: Any | None = None,
    ) -> np.ndarray:
        """Chain the frozen rights mask through the common convex projector."""

        restricted = self.restrict(raw_action)
        result = projection if projection is not None else self.inner.project(restricted, state)
        jacobian = self.inner.local_jacobian(restricted, state, projection=result)
        retained = np.ones(len(self.model.layout.keys), dtype=float)
        if not self.readiness_allowed:
            for key in (*self.model.layout.readiness_order, *self.model.layout.readiness_exercise):
                retained[self.model.layout.keys.index(key)] = 0.0
        if not self.direct_allowed:
            for key in self.model.layout.direct_order:
                retained[self.model.layout.keys.index(key)] = 0.0
        return jacobian @ np.diag(retained)


def configure_capacity_rights(model: BenchmarkModel, capacity_rights: str) -> CapacityRightsProjector:
    wrapper = CapacityRightsProjector(
        inner=model.projector,
        model=model,
        capacity_rights=capacity_rights,
    )
    model.projector = wrapper  # type: ignore[assignment]
    return wrapper


def load_actor(path: Path, expected_sha256: str) -> tuple[LinearActor, int]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Checkpoint hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        actor = LinearActor(
            np.asarray(data["weights"], dtype=float),
            np.asarray(data["log_standard_deviation"], dtype=float),
        )
        seed = int(data["training_seed"])
    return actor, seed


def _bundle_hash(
    controller: str,
    seed_index: int,
    bc_hash: str,
    sac_hash: str,
    regime: str,
    rights: str,
) -> str:
    payload = "|".join(
        [controller, str(seed_index), bc_hash, sac_hash, regime, rights]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_config(base_config: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["experiment_id"] = "5.2.4_symmetric_controller_training"
    return config


def train_condition(
    *,
    code_root: Path,
    benchmark_config_path: Path,
    information_regime: str,
    capacity_rights: str,
    checkpoint_root: Path,
    config_hash: str,
) -> dict[str, list[dict[str, Any]]]:
    """Worker-safe from-scratch training for one information/rights condition."""

    base_config = json.loads(benchmark_config_path.read_text(encoding="utf-8"))
    config = _training_config(base_config)
    frozen = load_frozen_5_2_1_inputs(config)
    hmm = load_hmm_inputs(frozen.output_dir)
    provider = InformationProvider(
        hmm=hmm,
        readiness_lead_weeks=int(config["action"]["readiness_lead_weeks"]),
    )
    model = build_model(config)
    configure_capacity_rights(model, capacity_rights)
    reference_normal = float(sum(model.gateway_scales.values()))
    training_paths, validation_paths = build_training_validation_paths(
        config=config,
        residuals=frozen.residuals,
        reference_normal_model_units=reference_normal,
    )
    training_paths = [provider.apply_training_regime(path, information_regime) for path in training_paths]
    validation_paths = [provider.apply_training_regime(path, information_regime) for path in validation_paths]
    teacher, teacher_hash = generate_teacher_data(model=model, paths=training_paths)
    condition = controller_id(information_regime, capacity_rights)
    directory = checkpoint_root / condition
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    for item in teacher:
        teacher_rows.append(
            {
                "controller_id": condition,
                "information_regime": information_regime,
                "capacity_rights": capacity_rights,
                "path_id": item.path_id,
                "period_offset": item.period_offset,
                "candidate_id": item.candidate_id,
                "nested_formal_objective": item.nested_objective,
                "teacher_hash": teacher_hash,
            }
        )
    for seed_index in range(int(config["training"]["seeds"])):
        bc = train_bc(
            model=model,
            teacher=teacher,
            teacher_hash=teacher_hash,
            validation_paths=validation_paths,
            seed_index=seed_index,
        )
        sac = train_sac(
            model=model,
            training_paths=training_paths,
            validation_paths=validation_paths,
            seed_index=seed_index,
            constrained=True,
        )
        bc_path, bc_hash = save_checkpoint(
            result=bc,
            directory=directory,
            feature_names=state_feature_names(model),
            config_hash=config_hash,
        )
        # save_checkpoint uses the policy name; keep both files in the same condition directory.
        sac_path, sac_hash = save_checkpoint(
            result=sac,
            directory=directory,
            feature_names=state_feature_names(model),
            config_hash=config_hash,
        )
        bundle_hash = _bundle_hash(
            condition,
            seed_index,
            bc_hash,
            sac_hash,
            information_regime,
            capacity_rights,
        )
        bundle_path = directory / f"controller_bundle_seed_{seed_index}.json"
        bundle_path.write_text(
            json.dumps(
                {
                    "controller_id": condition,
                    "information_regime": information_regime,
                    "capacity_rights": capacity_rights,
                    "seed_index": seed_index,
                    "training_seed": sac.seed,
                    "bc_checkpoint_sha256": bc_hash,
                    "constrained_sac_checkpoint_sha256": sac_hash,
                    "controller_bundle_sha256": bundle_hash,
                    "generated_from_scratch": True,
                    "test_paths_seen": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "controller_id": condition,
                "information_regime": information_regime,
                "capacity_rights": capacity_rights,
                "seed_index": seed_index,
                "training_seed": sac.seed,
                "bc_checkpoint_path": bc_path.relative_to(code_root).as_posix(),
                "bc_checkpoint_sha256": bc_hash,
                "sac_checkpoint_path": sac_path.relative_to(code_root).as_posix(),
                "sac_checkpoint_sha256": sac_hash,
                "controller_bundle_path": bundle_path.relative_to(code_root).as_posix(),
                "controller_bundle_sha256": bundle_hash,
                "bc_selected_episode": bc.selected_episode,
                "sac_selected_episode": sac.selected_episode,
                "bc_validation_loss": bc.best_validation_loss,
                "sac_validation_loss": sac.best_validation_loss,
                "generated_from_scratch": True,
                "reused_5_2_2_anchor": False,
                "test_paths_seen_before_selection": False,
                "teacher_hash": teacher_hash,
            }
        )
        for record in bc.training_curve + sac.training_curve:
            training_rows.append(
                {
                    "controller_id": condition,
                    "information_regime": information_regime,
                    "capacity_rights": capacity_rights,
                    **record,
                }
            )
        for record in bc.validation_curve + sac.validation_curve:
            validation_rows.append(
                {
                    "controller_id": condition,
                    "information_regime": information_regime,
                    "capacity_rights": capacity_rights,
                    **record,
                }
            )
    return {
        "manifest": rows,
        "training": training_rows,
        "validation": validation_rows,
        "teacher": teacher_rows,
    }


def copy_frozen_il_rd_controller(
    *,
    code_root: Path,
    benchmark_output: Path,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    manifest = pd.read_csv(benchmark_output / "checkpoint_manifest.csv")
    bc_rows = manifest.loc[manifest["policy"].eq("Behaviour cloning")].set_index("seed_index")
    sac_rows = manifest.loc[manifest["policy"].eq("Constrained SAC")].set_index("seed_index")
    target = checkpoint_root / "IL_RD"
    target.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed_index in sorted(set(bc_rows.index) & set(sac_rows.index)):
        bc = bc_rows.loc[seed_index]
        sac = sac_rows.loc[seed_index]
        source_bc = benchmark_output / str(bc["checkpoint_path"])
        source_sac = benchmark_output / str(sac["checkpoint_path"])
        if sha256_file(source_bc) != str(bc["checkpoint_sha256"]):
            raise RuntimeError("Frozen 5.2.2 BC checkpoint hash mismatch")
        if sha256_file(source_sac) != str(sac["checkpoint_sha256"]):
            raise RuntimeError("Frozen 5.2.2 SAC checkpoint hash mismatch")
        target_bc = target / f"behaviour_cloning_seed_{seed_index}.npz"
        target_sac = target / f"constrained_sac_seed_{seed_index}.npz"
        shutil.copy2(source_bc, target_bc)
        shutil.copy2(source_sac, target_sac)
        bc_hash = sha256_file(target_bc)
        sac_hash = sha256_file(target_sac)
        bundle_hash = _bundle_hash("IL_RD", int(seed_index), bc_hash, sac_hash, "IL", "RD")
        bundle_path = target / f"controller_bundle_seed_{seed_index}.json"
        bundle_path.write_text(
            json.dumps(
                {
                    "controller_id": "IL_RD",
                    "information_regime": "IL",
                    "capacity_rights": "RD",
                    "seed_index": int(seed_index),
                    "training_seed": int(sac["training_seed"]),
                    "bc_checkpoint_sha256": bc_hash,
                    "constrained_sac_checkpoint_sha256": sac_hash,
                    "controller_bundle_sha256": bundle_hash,
                    "generated_from_scratch": False,
                    "source": "accepted 5.2.2 proposed policy checkpoint pair",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "controller_id": "IL_RD",
                "information_regime": "IL",
                "capacity_rights": "RD",
                "seed_index": int(seed_index),
                "training_seed": int(sac["training_seed"]),
                "bc_checkpoint_path": target_bc.relative_to(code_root).as_posix(),
                "bc_checkpoint_sha256": bc_hash,
                "sac_checkpoint_path": target_sac.relative_to(code_root).as_posix(),
                "sac_checkpoint_sha256": sac_hash,
                "controller_bundle_path": bundle_path.relative_to(code_root).as_posix(),
                "controller_bundle_sha256": bundle_hash,
                "bc_selected_episode": int(bc["selected_episode"]),
                "sac_selected_episode": int(sac["selected_episode"]),
                "bc_validation_loss": float(bc["best_validation_operational_loss"]),
                "sac_validation_loss": float(sac["best_validation_operational_loss"]),
                "generated_from_scratch": False,
                "reused_5_2_2_anchor": True,
                "test_paths_seen_before_selection": False,
                "teacher_hash": str(bc.get("teacher_action_hash", "")),
            }
        )
    if len(rows) != 3:
        raise RuntimeError("The accepted 5.2.2 proposed policy does not have three seeds")
    return rows


def build_policies(
    *,
    code_root: Path,
    model: BenchmarkModel,
    controller_rows: pd.DataFrame,
) -> list[ModelGuidedPolicy]:
    policies: list[ModelGuidedPolicy] = []
    for row in controller_rows.sort_values("seed_index").itertuples(index=False):
        bc, _ = load_actor(code_root / row.bc_checkpoint_path, row.bc_checkpoint_sha256)
        sac, seed = load_actor(code_root / row.sac_checkpoint_path, row.sac_checkpoint_sha256)
        policies.append(
            ModelGuidedPolicy(
                model=model,
                bc_actor=bc,
                sac_actor=sac,
                training_seed=seed,
            )
        )
    return policies


def capacity_rights_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "capacity_rights": key,
                "readiness_allowed": values[0],
                "direct_procurement_allowed": values[1],
                "release_pacing_retained": True,
                "disclosure_retained": True,
                "restriction_location": "after raw action and before common projector",
            }
            for key, values in RIGHTS.items()
        ]
    )
