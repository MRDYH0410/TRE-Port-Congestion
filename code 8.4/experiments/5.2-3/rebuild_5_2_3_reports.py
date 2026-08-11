"""Rebuild/finalise 5.2.3 reports and manifest without rerunning policy trajectories."""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from reporting_5_2_3 import create_figures, write_reports, write_run_manifest  # noqa: E402


def main() -> int:
    started = time.perf_counter()
    config_path = EXPERIMENT_DIR / "config_5_2_3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    final = CODE_ROOT / str(config["output_directory"])
    staging = final.parent / f".{final.name}.staging"
    source = staging if staging.exists() else final
    if not source.exists():
        raise FileNotFoundError("No completed 5.2.3 tables are available to rebuild")
    policy_set = pd.read_csv(source / "mechanism_policy_set.csv")
    full_summary = pd.read_csv(source / "full_policy_mechanism_summary.csv")
    activation = pd.read_csv(source / "proposed_policy_activation_audit.csv")
    restricted = pd.read_csv(source / "restricted_action_paired_effects.csv")
    weekly = pd.read_csv(source / "weekly_policy_mechanisms.csv")
    physical = pd.read_csv(source / "physical_tagged_trajectory.csv")
    medoid = pd.read_csv(source / "path_medoid_selection.csv")
    acceptance = json.loads((source / "acceptance_5_2_3.json").read_text(encoding="utf-8"))
    leader = str(policy_set.loc[policy_set["is_benchmark_leader"].astype(bool), "policy"].iloc[0])
    figure_policies = policy_set.loc[policy_set["formal_figure_policy"].astype(bool), "policy"].tolist()
    medoid_path = str(medoid.loc[medoid["selected_physical_path_medoid"].astype(bool), "path_id"].iloc[0])
    create_figures(
        weekly=weekly,
        physical=physical,
        restricted_effects=restricted,
        figure_policies=figure_policies,
        medoid_path_id=medoid_path,
        output_directory=source / "figures",
        dpi=int(config["figures"]["dpi"]),
    )
    report_directory = (CODE_ROOT / str(config["report_directory"])).resolve()
    report_paths = write_reports(
        report_directory=report_directory,
        policy_set=policy_set,
        benchmark_leader=leader,
        figure_policies=figure_policies,
        medoid_path_id=medoid_path,
        full_summary=full_summary,
        activation=activation,
        restricted_effects=restricted,
        acceptance=acceptance,
        weekly=weekly,
        physical=physical,
    )
    if source == staging:
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    public = CODE_ROOT / str(config["figure_directory"])
    public.mkdir(parents=True, exist_ok=True)
    for figure in (final / "figures").glob("*.png"):
        shutil.copy2(figure, public / figure.name)
    output_files = [
        *final.glob("*.csv"),
        final / "acceptance_5_2_3.json",
        *(final / "figures").glob("*.png"),
        *public.glob("*.png"),
        *report_paths,
    ]
    benchmark = CODE_ROOT / str(config["input_5_2_2"])
    manifest = write_run_manifest(
        output_directory=final,
        code_root=CODE_ROOT,
        config_path=config_path,
        input_files=[
            benchmark / "acceptance_5_2_2.json",
            benchmark / "run_manifest.json",
            benchmark / "path_level_seed_aggregated.csv",
            benchmark / "benchmark_period_paths.csv",
            benchmark / "requested_and_implemented_actions.csv",
            benchmark / "benchmark_replications.csv",
            benchmark / "checkpoint_manifest.csv",
            benchmark / "test_path_manifest.csv",
        ],
        output_files=output_files,
        started_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        status=str(acceptance["status"]),
    )
    print(json.dumps({"status": acceptance["status"], "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
