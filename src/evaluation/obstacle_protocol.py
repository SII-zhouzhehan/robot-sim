"""Geometry-independent acceptance rules for isolated obstacle traversal."""
from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class TraversalCriteria:
    goal_tolerance_m: float = 0.2
    clearance_margin_m: float = 0.02
    joint_tolerance_rad: float = 0.05
    settle_steps: int = 10
    penetration_tolerance_m: float = 0.002
    contact_force_tolerance_n: float = 1.0


class TraversalMonitor:
    def __init__(self, entrance_x, exit_x, center_y, half_lane, criteria=None):
        self.criteria = criteria or TraversalCriteria()
        self.entrance_x = entrance_x
        self.exit_x = exit_x
        self.center_y = center_y
        self.half_lane = half_lane
        self.approached = False
        self.crossed = False
        self.failure = None
        self.stable_steps = 0

    def observe(self, *, min_x, max_x, center_y, goal_distance,
                joint_error, morphology_reached, supported,
                collision=False, invalid_state=False):
        if invalid_state:
            self.failure = self.failure or "invalid_state"
        if collision:
            self.failure = self.failure or "obstacle_contact"
        if max_x < self.entrance_x:
            self.approached = True
        overlapping = max_x >= self.entrance_x and min_x <= self.exit_x
        if overlapping and abs(center_y-self.center_y) > self.half_lane:
            self.failure = self.failure or "bypassed_corridor"
        if self.approached and min_x > self.exit_x + self.criteria.clearance_margin_m:
            self.crossed = True
        acceptable = (not self.failure and self.crossed and supported
                      and morphology_reached
                      and goal_distance < self.criteria.goal_tolerance_m
                      and joint_error <= self.criteria.joint_tolerance_rad)
        self.stable_steps = self.stable_steps + 1 if acceptable else 0
        return self.stable_steps >= self.criteria.settle_steps


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return [0.0, 1.0]
    proportion = successes / trials
    denom = 1 + z*z/trials
    center = (proportion + z*z/(2*trials)) / denom
    radius = z * sqrt(proportion*(1-proportion)/trials + z*z/(4*trials*trials)) / denom
    return [max(0.0, center-radius), min(1.0, center+radius)]
