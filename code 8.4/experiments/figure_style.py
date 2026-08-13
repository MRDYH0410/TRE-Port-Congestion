"""Shared publication style and visible-text safeguards for Chapter 5 figures.

This module is intentionally presentation only.  It does not read experimental
inputs, transform reported values, or participate in simulation and inference.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.text import Text


TEXT_WIDTH = 7.45
INK = "#263238"
MUTED = "#5F6B73"
GRID = "#D9DEE5"
BLUE = "#0072B2"
BLUE_LIGHT = "#DCE8F2"
ORANGE = "#D97925"
TEAL = "#009E73"
GOLD = "#E69F00"
PURPLE = "#8E6C8A"
GREY = "#6B7280"

POLICY_LABELS = {
    "Passive": "Passive",
    "Reactive": "Reactive",
    "Projected stochastic MPC": "Projected MPC",
    "Behaviour cloning": "Behaviour cloning",
    "PPO": "PPO",
    "Vanilla SAC": "Standard SAC",
    "Constrained SAC": "Constrained SAC",
    "Model-guided constrained SAC": "Model guided SAC",
    "Conventional MSA": "Conventional MSA",
    "RC-MSA": "RC MSA",
    "Projected stochastic MPC controlled audit": "Projected MPC audit",
}

POLICY_COLOURS = {
    "Passive": GREY,
    "Reactive": BLUE,
    "Projected stochastic MPC": GOLD,
    "Behaviour cloning": TEAL,
    "PPO": ORANGE,
    "Vanilla SAC": "#56B4E9",
    "Constrained SAC": "#332288",
    "Model-guided constrained SAC": "#CC79A7",
}

POLICY_MARKERS = {
    "Passive": "o",
    "Reactive": "s",
    "Projected stochastic MPC": "^",
    "Behaviour cloning": "D",
    "PPO": "v",
    "Vanilla SAC": "P",
    "Constrained SAC": "X",
    "Model-guided constrained SAC": "h",
}

POLICY_LINESTYLES = {
    "Passive": ":",
    "Reactive": "-",
    "Projected stochastic MPC": "--",
    "Behaviour cloning": "-.",
    "PPO": (0, (3, 1, 1, 1)),
    "Vanilla SAC": (0, (5, 2)),
    "Constrained SAC": (0, (2, 1)),
    "Model-guided constrained SAC": (0, (5, 1, 1, 1)),
}

WARNING_LABELS = {
    "GH": "Historical release",
    "GT": "Timely release",
    "GL": "Late release",
    "GFW": "False warning",
}

CAPACITY_RIGHT_LABELS = {
    "RD": "Both rights",
    "R": "Readiness only",
    "D": "Direct only",
    "NONE": "Neither right",
}

INFORMATION_COMPARISON_LABELS = {
    "IF vs I0": "Current risk information",
    "IL vs I0": "Lead risk forecast",
    "ORACLE vs I0": "Perfect information",
}

CAPACITY_EFFECT_LABELS = {
    "V_R_given_D": "Readiness value",
    "V_D_given_R": "Direct capacity value",
    "S_RD": "Joint effect",
}

CONTRACT_LABELS = {
    "M0_UPSTREAM_LOCKS": "Upstream evidence locked",
    "M1_RELEASED_INFORMATION": "Released risk information",
    "M2_MATCHED_SCENARIOS": "Matched policy scenarios",
    "M3_FEASIBLE_PROJECTION": "Feasible joint projection",
    "M4_RCMSA_EQUILIBRIUM": "Behavioural equilibrium",
    "M5_TAGGED_TRANSITION": "Tagged network transition",
    "M6_COMPLETE_LOSS": "Complete loss accounting",
    "M7_NONANTICIPATIVITY": "Nonanticipative decisions",
    "M8_NESTED_MPC": "Nested MPC rollout",
    "M9_BC_SAC_SELECTOR": "BC and SAC selection",
    "M10_TRAVEL_LAG": "Travel lag preservation",
    "M11_CAPACITY_TIMING": "Capacity delivery timing",
    "M12_REPRODUCIBILITY": "Reproducible aggregates",
    "M13_SAC_LATENT_GAUSSIAN": "Latent Gaussian sampling",
    "M14_SAC_ACTOR_MEAN_UPDATE": "Actor mean update",
    "M15_SAC_LOG_STD_UPDATE": "Actor variance update",
    "M16_SAC_ENTROPY_ACTOR_TERM": "Entropy in actor objective",
    "M17_SAC_ENTROPY_TEMPERATURE": "Adaptive entropy temperature",
    "M18_SAC_TWIN_REWARD_CRITICS": "Twin reward critics",
    "M19_SAC_CONSTRAINT_CRITIC": "Constraint critic",
    "M20_SAC_CONSTRAINT_DUAL": "Constraint dual",
    "M21_SAC_PROJECTION_GRADIENT": "Projection gradient",
    "M22_SAC_FINITE_DIFFERENCE": "Finite difference gradient",
    "M23_VALIDATION_CHECKPOINT": "Validation checkpoint selection",
    "M24_CHECKPOINT_REPLAY": "Checkpoint replay",
    "M25_UNAVAILABLE_ROUTE_HOLD": "Unavailable route holding",
    "M26_CLEARANCE_TERMINAL": "Clearance and terminal loss",
    "M27_RCMSA_MASTER_CHOICE_DISTANCE": "Full choice support",
    "M28_DISCLOSURE_REFERENCE_ACTION": "Disclosure reference isolation",
    "M29_WAITING_VINTAGE_NO_RESET": "Waiting vintage preservation",
    "M30_MPC_SELECTOR_MODULE_CERTIFICATES": "Nested module certificates",
    "M31_CORE_REPAIR_NUMERICAL_EQUIVALENCE": "Numerical equivalence",
}

FACTOR_LABELS = {
    "route_sensitivity": "Route choice sensitivity",
    "waiting_hazard_power": "Waiting age sensitivity",
    "exit_consequence": "Exit consequence",
    "network_exposure": "Network exposure",
    "maritime_lag": "Maritime lag",
    "port_service": "Port service",
    "corridor_capacity": "Corridor capacity",
    "physical_feedback": "Congestion feedback",
    "route_resource_cost": "Route resource cost",
    "action_cost": "Action cost",
    "authority_budget": "Authority budget",
    "readiness_lead": "Readiness lead",
    "waiting_error_scale": "Waiting forecast error",
    "information_credibility": "Information credibility",
}

INTERACTION_LABELS = {
    "interaction__long_lag__severe_reclosure": "Long maritime lag and severe reclosure",
    "interaction__convex_hazard__low_exit": "Convex waiting hazard and low exit consequence",
    "interaction__n09__low_corridor": "Nine gateways and low corridor capacity",
    "interaction__lead16__low_credibility": "Long readiness lead and low information credibility",
}


def apply_publication_style() -> None:
    """Apply a legible style calibrated to the manuscript text width."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.7,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.fontsize": 7.6,
            "legend.frameon": False,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def panel_title(axis: Any, letter: str, title: str) -> None:
    axis.set_title(f"{letter}. {title}", loc="left", pad=5)


def policy_label(policy: str) -> str:
    return POLICY_LABELS.get(policy, str(policy).replace("_", " "))


def factor_label(cell_id: str, factor: str, level: Any) -> str:
    """Translate robustness implementation fields into short domain labels."""

    if cell_id == "reference":
        return "Reference"
    if cell_id in INTERACTION_LABELS:
        return INTERACTION_LABELS[cell_id]
    name = FACTOR_LABELS.get(str(factor), str(factor).replace("_", " ").title())
    numeric = float(level)
    if factor == "readiness_lead":
        value = f"{numeric:g} weeks"
    elif factor == "waiting_hazard_power":
        value = f"power {numeric:g}"
    elif factor == "physical_feedback" and numeric == 0:
        value = "off"
    else:
        value = f"{numeric:g} times"
    return f"{name}  {value}"


_FORBIDDEN_VISIBLE_TEXT = (
    re.compile(r"\bFigure\s+\d", re.IGNORECASE),
    re.compile(r"\btest_[A-Za-z0-9_]+", re.IGNORECASE),
    re.compile(r"\bseed(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bM\d+_[A-Z0-9_]+\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
)


def assert_publication_text(fig: Any) -> None:
    """Fail before export when implementation language leaks into a figure."""

    violations: list[str] = []
    for item in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        if not item.get_visible():
            continue
        value = item.get_text().strip()
        if not value:
            continue
        if any(pattern.search(value) for pattern in _FORBIDDEN_VISIBLE_TEXT):
            violations.append(value)
    if violations:
        unique = list(dict.fromkeys(violations))
        raise ValueError("Nonpublication text in figure: " + " | ".join(unique))


def save_figure(
    fig: Any,
    path: Path,
    *,
    dpi: int = 300,
    metadata: dict[str, Any] | None = None,
) -> None:
    assert_publication_text(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
