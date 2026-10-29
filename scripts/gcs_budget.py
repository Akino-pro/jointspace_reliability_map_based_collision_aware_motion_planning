"""Deterministic stage-one quotas without dropping any goal target."""
from numbers import Integral


def allocate_target_budgets(target_count, total_paths, total_trials):
    values = (target_count, total_paths, total_trials)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in values):
        raise ValueError("Target count and total rounding budgets must be positive integers")
    if total_paths < target_count or total_trials < target_count:
        raise ValueError("Total budgets must allow at least one path and trial per target; no targets are dropped")
    if total_trials < total_paths:
        raise ValueError("Total trial budget must be at least the total candidate-path budget")
    path_base, path_remainder = divmod(total_paths, target_count)
    trial_base, trial_remainder = divmod(total_trials, target_count)
    return [dict(max_rounded_paths=path_base+int(i < path_remainder),
                 max_rounding_trials=trial_base+int(i < trial_remainder)) for i in range(target_count)]
