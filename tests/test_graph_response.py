"""Distinguish graph priors, measured relations and actuator commands."""
import numpy as np

from exploration.graph_response import ResponseModel, graph_penalty, response_graph_evidence


def test_graph_prior_keeps_nonadjacent_effects_possible():
    rng = np.random.default_rng(3)
    commands = rng.normal(size=(80, 5)) * .02
    truth = np.eye(5)
    truth[4, 0] = -.7
    penalty = graph_penalty(5, [(i, i + 1) for i in range(4)])
    model = ResponseModel.fit(commands, commands @ truth.T, penalty,
                              regularization=1e-5, scale=.02)
    assert penalty[4, 0] > penalty[1, 0] > penalty[0, 0]
    assert model.matrix[4, 0] < -.69


def test_graph_response_is_invariant_to_consistent_node_relabeling():
    rng = np.random.default_rng(11)
    commands, truth = rng.normal(size=(50, 6)) * .02, rng.normal(size=(6, 6))
    responses = commands @ truth.T
    penalty = graph_penalty(6, [(0, 1), (0, 3), (1, 2), (3, 4), (4, 5)])
    model = ResponseModel.fit(commands, responses, penalty)
    order = rng.permutation(6)
    permuted = ResponseModel.fit(commands[:, order], responses[:, order], penalty[np.ix_(order, order)])
    np.testing.assert_allclose(permuted.matrix, model.matrix[np.ix_(order, order)], atol=1e-12)


def test_motion_relation_is_not_a_command_ratio_with_unequal_servo_gains():
    # q2 = 2*q1, but q1=.1*u1+.4*u2. Output and input directions differ.
    response = np.array([[.1, .4], [.2, .8]])
    commands = np.eye(2) * .02
    model = ResponseModel.fit(commands, commands @ response.T, np.ones((2, 2)),
                              regularization=1e-10)
    desired = np.array([.004, .008])
    command = model.command_for_motion(desired, [-.02, -.02], [.02, .02], damping=1e-6)
    np.testing.assert_allclose(model.predict(command), desired, atol=1e-7)
    assert not np.allclose(command, desired)
    assert np.linalg.norm(command) <= .02


def test_underexcitation_never_creates_confirmed_constraints():
    commands = np.ones((30, 4)) * .01
    model = ResponseModel.fit(commands, commands, np.ones((4, 4)))
    evidence = response_graph_evidence(model, commands, commands, commands)
    assert evidence["status"] == "insufficient_input_excitation"
    assert evidence["candidate_motion_factors"] == []
    assert evidence["response_edges"] == []


def test_general_multijoint_relation_is_checked_on_independent_data():
    rng = np.random.default_rng(2)
    normal = np.array([1., 1., -1.]) / np.sqrt(3)
    truth = np.eye(3) - np.outer(normal, normal)
    commands = rng.normal(size=(60, 3)) * .02
    responses = commands @ truth.T
    heldout = rng.normal(size=(20, 3)) * .02 @ truth.T
    model = ResponseModel.fit(commands, responses, np.ones((3, 3)), regularization=1e-5)
    evidence = response_graph_evidence(model, commands, responses, heldout, bootstrap=40)
    factors = evidence["candidate_motion_factors"]
    assert len(factors) == 1
    assert factors[0]["members"] == [0, 1, 2]
    assert factors[0]["status"] == "validated_local_response_relation"
    # The same fitted relation must fail when new measurements contradict it.
    contradictory = rng.normal(size=(20, 3)) * .02
    evidence = response_graph_evidence(model, commands, responses, contradictory, bootstrap=40)
    assert evidence["candidate_motion_factors"][0]["status"] == "unconfirmed_local_relation"


def test_weak_single_joint_response_is_not_a_permanent_lock():
    rng = np.random.default_rng(9)
    commands = rng.normal(size=(50, 3)) * .02
    truth = np.diag([1., .01, 1.])
    responses = commands @ truth.T
    model = ResponseModel.fit(commands, responses, np.ones((3, 3)))
    evidence = response_graph_evidence(model, commands, responses, responses, bootstrap=40)
    assert evidence["candidate_motion_factors"] == []
    assert evidence["weak_response_directions"][0]["status"] == "weak_response_not_a_locked_joint"


def test_control_respects_one_sided_and_fixed_joint_bounds():
    model = ResponseModel.fit(np.eye(3) * .02, np.eye(3) * .02, np.ones((3, 3)))
    command = model.command_for_motion([.02, -.02, .02], [-.02, 0, 0], [0, .02, 0])
    assert command[0] <= 0
    assert command[1] >= 0
    assert command[2] == 0
    assert np.linalg.norm(command) < 1e-7
