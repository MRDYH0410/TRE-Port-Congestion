"""Domain-specific failures.  Numerical failure never fabricates a transition."""


class TREError(Exception):
    """Base class for model and algorithm errors."""


class ContractError(TREError):
    """An input violates a Chapter 3 or Chapter 4 contract."""


class NumericalFailure(TREError):
    """A numerical module failed its declared certificate."""


class InfeasibleAction(TREError):
    """An action does not belong to the hard feasible set."""


class AuditFailure(TREError):
    """A conservation, timing, or state-completeness audit failed."""

