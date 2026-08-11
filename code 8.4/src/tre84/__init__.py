"""TRE 8.4 model-consistent simulation and control framework.

The package implements the formal contracts in Chapters 3 and 4.  It contains
no Chapter 5 calibration, historical replay result, or experiment conclusion.
"""

from .actions import Action, ActionDomain, ActionKey, ActionProjector, Block
from .acceptance import PeriodInformation, evaluate_trajectory_acceptance
from .behavior import RCMSASettings, RCMSASolver, oldest_first
from .engine import ModelKernel
from .keys import Network, Provenance, ResourceKey, Route, SourceKey, Stage, Tag
from .inference import PrecisionRule, StudentInterval, holm_adjust, student_interval
from .metrics import TrajectoryStatistics, compute_trajectory_statistics
from .scenarios import RevealedEventHistory, TimestampedOperationalContext
from .state import ModelState

__all__ = [
    "Action",
    "ActionDomain",
    "ActionKey",
    "ActionProjector",
    "Block",
    "ModelKernel",
    "ModelState",
    "Network",
    "Provenance",
    "PrecisionRule",
    "PeriodInformation",
    "RCMSASettings",
    "RCMSASolver",
    "ResourceKey",
    "RevealedEventHistory",
    "Route",
    "SourceKey",
    "Stage",
    "StudentInterval",
    "Tag",
    "TimestampedOperationalContext",
    "TrajectoryStatistics",
    "compute_trajectory_statistics",
    "evaluate_trajectory_acceptance",
    "holm_adjust",
    "oldest_first",
    "student_interval",
]
