import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.obstacle_protocol import TraversalMonitor, wilson_interval


class TraversalTests(unittest.TestCase):
    def setUp(self):
        self.monitor = TraversalMonitor(1.9, 2.1, 0., .21)
        self.state = dict(min_x=0., max_x=.5, center_y=0., goal_distance=4.,
                          joint_error=0., morphology_reached=True, supported=True)
        self.monitor.observe(**self.state)

    def finish(self, **changes):
        state = dict(self.state, min_x=3.4, max_x=4.2, goal_distance=.1)
        state.update(changes)
        return [self.monitor.observe(**state) for _ in range(10)][-1]

    def test_goal_requires_complete_body_clearance(self):
        self.assertFalse(self.finish(min_x=2.0))

    def test_success_requires_settled_actual_morphology_and_support(self):
        self.assertFalse(self.finish(joint_error=.2))
        self.assertFalse(self.finish(morphology_reached=False))
        self.assertFalse(self.finish(supported=False))
        self.assertTrue(self.finish())

    def test_collision_is_sticky_even_after_goal_arrival(self):
        self.monitor.observe(**dict(self.state, collision=True))
        self.assertFalse(self.finish())

    def test_bypass_is_not_counted_as_success(self):
        self.monitor.observe(**dict(self.state, min_x=1.8, max_x=2.3, center_y=.5))
        self.assertFalse(self.finish())

    def test_spawn_after_obstacle_is_not_a_traversal(self):
        self.monitor = TraversalMonitor(1.9, 2.1, 0., .21)
        self.assertFalse(self.finish())

    def test_twenty_successes_do_not_prove_95_percent_population_rate(self):
        lower, upper = wilson_interval(20, 20)
        self.assertLess(lower, .95)
        self.assertAlmostEqual(upper, 1.)


if __name__ == "__main__":
    unittest.main()
