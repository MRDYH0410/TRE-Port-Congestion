"""Freeze route-wise waiting forecast error scales before 5.2.2 training.

The calibration uses only the two designed validation paths and zero
coordination actions.  Disclosure intensity is therefore zero, so the
temporary scale used to execute this diagnostic cannot affect the SUE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(CODE_ROOT / "src"))

from model import _current_route_wait, build_model  # noqa: E402
from paths import build_training_validation_paths, load_frozen_5_2_1_inputs  # noqa: E402
from simulator import build_realization  # noqa: E402


def main() -> int:
    config_path = EXPERIMENT_DIR / "config_5_2_2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = load_frozen_5_2_1_inputs(config)
    model = build_model(config)
    _, validation_paths = build_training_validation_paths(
        config=config,
        residuals=frozen.residuals,
        reference_normal_model_units=float(sum(model.gateway_scales.values())),
    )
    records: list[dict[str, object]] = []
    for path in validation_paths:
        rows = [row._asdict() for row in path.frame.itertuples(index=False)]
        state = model.initial_state(rows[0])
        for offset, row in enumerate(rows[:-1]):
            forecasts = {
                route_id: _current_route_wait(state, model, route_id)
                for route_id in sorted(model.network.routes)
            }
            action = model.zero_action()
            projection = model.projector.project(action, state)
            realization = build_realization(
                model=model,
                state=state,
                row=row,
                next_row=rows[offset + 1],
            )
            result = model.kernel.execute(
                state=state,
                action=projection.action,
                realization=realization,
                projection=projection,
            )
            next_state = result.transition.next_state
            for route_id, forecast in forecasts.items():
                realised = _current_route_wait(next_state, model, route_id)
                records.append(
                    {
                        "split": "validation",
                        "path_id": path.path_id,
                        "origin_week": row["week"],
                        "target_week": rows[offset + 1]["week"],
                        "route": route_id,
                        "reference_wait_forecast_weeks": forecast,
                        "realised_next_state_wait_proxy_weeks": realised,
                        "forecast_error_weeks": realised - forecast,
                        "absolute_error_weeks": abs(realised - forecast),
                        "squared_error_weeks2": (realised - forecast) ** 2,
                    }
                )
            state = next_state
    residuals = pd.DataFrame(records)
    summary = (
        residuals.groupby("route", as_index=False)
        .agg(
            validation_errors=("forecast_error_weeks", "size"),
            mean_error_weeks=("forecast_error_weeks", "mean"),
            mae_weeks=("absolute_error_weeks", "mean"),
            mean_squared_error_weeks2=("squared_error_weeks2", "mean"),
        )
        .sort_values("route")
    )
    summary["sigma_W_rmse_weeks"] = np.sqrt(summary["mean_squared_error_weeks2"])
    summary["estimator"] = "sqrt(mean(one-step forecast error squared))"
    summary["information_cutoff"] = "validation paths only; frozen before teacher generation and policy training"
    summary["policy"] = "Passive zero coordination action"
    summary["uses_historical_test_event"] = False
    if (summary["sigma_W_rmse_weeks"] <= 0).any() or summary["sigma_W_rmse_weeks"].isna().any():
        raise RuntimeError("Waiting forecast error scales must be positive and estimable")
    residuals.to_csv(
        EXPERIMENT_DIR / "waiting_forecast_error_residuals.csv", index=False, lineterminator="\n"
    )
    summary.to_csv(
        EXPERIMENT_DIR / "waiting_forecast_error_calibration.csv", index=False, lineterminator="\n"
    )
    print(summary[["route", "validation_errors", "sigma_W_rmse_weeks"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
