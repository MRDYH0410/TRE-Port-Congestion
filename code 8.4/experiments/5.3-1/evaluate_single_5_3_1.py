"""Pre-evaluate one late 5.3.1 grid cell using an isolated path cache."""

from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
for entry in (
    EXPERIMENT_DIR,
    CODE_ROOT / "experiments" / "5.2-2",
    CODE_ROOT / "experiments" / "5.2-3",
    CODE_ROOT / "src",
):
    sys.path.insert(0, str(entry))

from model import build_model  # noqa: E402
from paths import (  # noqa: E402
    build_test_paths,
    build_training_validation_paths,
    load_frozen_5_2_1_inputs,
)
from run_5_3_1 import (  # noqa: E402
    _evaluate_cell,
    _json_hash,
    _load_configs,
    _model_config,
    _policy_instances,
    _source_bundle_hash,
    _train_cell,
    _verify_upstream,
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: evaluate_single_5_3_1.py CHI WORKERS")
    chi = float(sys.argv[1])
    workers = int(sys.argv[2])
    experiment, base, config_hash = _load_configs()
    if chi not in [float(value) for value in experiment["commitment_grid"]]:
        raise ValueError(f"chi={chi} is outside the frozen grid")
    upstream = _verify_upstream(experiment)
    signature = _json_hash(
        {
            "config_hash": config_hash,
            "source_hash": _source_bundle_hash(),
            "upstream": upstream[["relative_path", "observed_sha256"]].to_dict(
                orient="records"
            ),
        }
    )
    frozen = load_frozen_5_2_1_inputs(base)
    reference = build_model(_model_config(base, experiment, 0.5))
    training, validation = build_training_validation_paths(
        config=base,
        residuals=frozen.residuals,
        reference_normal_model_units=float(sum(reference.gateway_scales.values())),
    )
    model_config = _model_config(base, experiment, chi)
    model = build_model(model_config)
    results, _ = _train_cell(
        chi=chi,
        model=model,
        training_paths=training,
        validation_paths=validation,
        run_signature=signature,
    )
    paths = build_test_paths(config=base, frozen=frozen, count=88)
    _evaluate_cell(
        chi=chi,
        model=model,
        model_config=model_config,
        policies=_policy_instances(model, results),
        paths=paths,
        run_signature=signature,
        workers=workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
