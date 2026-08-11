from __future__ import annotations

import numpy as np

from tre84.inference import PrecisionRule, holm_adjust, student_interval


def test_precision_rule_selects_paths_from_pilot_variance() -> None:
    rule = PrecisionRule(
        target_halfwidth=1.0,
        confidence_level=0.95,
        minimum_paths=4,
        maximum_paths=200,
    )
    result = rule.required_paths(3.0)
    assert result.achieved_within_cap
    assert 4 <= result.required_paths <= 200
    assert result.halfwidth_at_required <= 1.0


def test_student_interval_and_holm_adjustment_are_family_aware() -> None:
    ordinary = student_interval([1.0, 2.0, 3.0, 4.0], confidence_level=0.95)
    simultaneous = student_interval(
        [1.0, 2.0, 3.0, 4.0], confidence_level=0.95, family_size=3
    )
    assert simultaneous.lower < ordinary.lower
    assert simultaneous.upper > ordinary.upper
    adjusted = holm_adjust([0.01, 0.03, 0.20])
    assert np.all(adjusted >= np.array([0.01, 0.03, 0.20]))
    assert np.all((0 <= adjusted) & (adjusted <= 1))
