"""Identification invariants: excitation, general group sizes and non-pair constraints."""
import numpy as np
import pytest

from exploration import ActiveMotionLearner


def identify(response, edges=()):
    n = len(response)
    learner = ActiveMotionLearner([f"motor{i}" for i in range(n)], edges, ridge=1e-10)
    for command in np.eye(n) * .1:
        learner.observe(command, response @ command)
    return learner


def test_shared_command_is_not_evidence_of_mechanical_coupling():
    learner = ActiveMotionLearner(["a", "b", "c"])
    for magnitude in (.01, .02, .03):
        command = np.ones(3) * magnitude
        learner.observe(command, command)
    assert learner.command_rank == 1
    assert learner.observed_rank is None
    assert learner.graph()["coupled_groups"] == []
    assert learner.graph()["linear_factors"] == []


@pytest.mark.parametrize("count", [1, 4, 7, 11])
def test_independent_motor_count_is_inferred(count):
    learner = identify(np.eye(count))
    assert learner.observed_rank == count
    assert learner.factors == []
    assert learner.constraints.shape == (0, count)
    assert learner.graph()["response_edges"] == []


def test_three_groups_with_different_sizes_and_nonlocal_members():
    # A black-box response with 2-, 3- and 4-motor groups, not two fixed pairs.
    groups = [[0, 5], [1, 4, 7], [2, 3, 6, 8]]
    B = np.zeros((9, 3))
    for column, group in enumerate(groups):
        B[group, column] = np.linspace(.5, 1.5, len(group)) * (-1) ** np.arange(len(group))
    response = B @ np.linalg.pinv(B)
    learner = identify(response, edges=[(i, i + 1) for i in range(8)])
    assert learner.observed_rank == 3
    assert {frozenset(f["members"]) for f in learner.factors} == {frozenset(g) for g in groups}
    candidates = learner.candidates(-np.ones(9), np.ones(9), .04)
    group_commands = [c for c in candidates if c.kind == "learned_graph_group"]
    assert {frozenset(c.members) for c in group_commands} == {frozenset(g) for g in groups}
    for command in group_commands:
        np.testing.assert_allclose(learner.constraints @ command.increment, 0, atol=1e-8)


def test_general_three_joint_relation_need_not_form_proportional_pairs():
    # q3 = q1 + q2; all three response rows are different directions.
    normal = np.array([1., 1., -1.])
    response = np.eye(3) - np.outer(normal, normal) / (normal @ normal)
    learner = identify(response)
    assert learner.observed_rank == 2
    assert learner.factors == []
    assert len(learner.graph()["linear_factors"]) == 1
    assert learner.graph()["linear_factors"][0]["members"] == [0, 1, 2]
    command = learner.coordinated_request([1, 0, 0], -np.ones(3), np.ones(3), .04)
    assert command is not None
    assert abs(normal @ command) < 1e-8


def test_limit_handling_preserves_a_whole_coordinated_direction():
    direction = np.array([1., -2., 1.])
    bounded = ActiveMotionLearner.bounded_direction(direction, [-1, -.01, -1], [.1, 1, .1], .1)
    assert bounded is not None
    np.testing.assert_allclose(bounded / direction, np.ones(3) * .005)
    assert ActiveMotionLearner.bounded_direction(direction, [-1, -1, -1], [0, 1, 1], .1) is None


def test_initial_joint_limits_do_not_require_two_sided_excitation():
    learner = ActiveMotionLearner(["a", "b", "c", "d"], [(0, 1), (1, 2), (2, 3)])
    lower, upper = np.array([-1, -1, 0, -1]), np.array([0, 0, 1, 0])
    for _ in range(4):
        candidate, _ = learner.choose(lower, upper, .04)
        assert np.all(candidate.increment >= lower)
        assert np.all(candidate.increment <= upper)
        learner.observe(candidate.increment, candidate.increment, kind=candidate.kind)
    assert learner.command_rank == 4
    assert learner.observed_rank == 4


def test_information_only_ablation_removes_graph_generated_probes():
    learner = identify(np.eye(4), edges=[(0, 1), (1, 2), (2, 3)])
    graph_action, _ = learner.choose(-np.ones(4), np.ones(4), .04)
    assert graph_action.kind == "physical_graph_pair"
    flat_action, _ = learner.choose(-np.ones(4), np.ones(4), .04, use_graph=False)
    assert flat_action.kind not in ("physical_graph_pair", "learned_graph_group")
