"""Checks for graph search information boundaries and general-dimensional actions."""
import numpy as np
import pytest

from exploration.graph_search import candidate_commands, fit_response, response_relations, choose_command


@pytest.mark.parametrize("n", [3, 8, 11])
def test_candidates_explore_multiple_joints_before_any_model(n):
    lo, hi = np.full(n, -.03), np.full(n, .03)
    hi[0] = 0
    pool, kinds = candidate_commands(lo, hi, .03, np.random.default_rng(8))
    assert "multi" in kinds and "pair" in kinds
    assert np.max(np.linalg.norm(pool, axis=1)) <= .030000001
    assert np.all(pool >= lo - 1e-12) and np.all(pool <= hi + 1e-12)
    assert np.linalg.matrix_rank(pool) == n


def test_learned_candidates_use_input_not_output_singular_vectors():
    # Different actuator gains mean output coupling is NOT the correct command direction.
    matrix = np.array([[.1, .4], [.2, .8]])
    pool, kinds = candidate_commands(np.full(2, -.03), np.full(2, .03), .03,
                                      np.random.default_rng(8), matrix=matrix)
    learned = pool[[k == "learned_command_direction" for k in kinds]]
    assert len(learned) == 2
    np.testing.assert_allclose(abs(learned[:, 1] / learned[:, 0]), 4, atol=1e-8)


def test_weld_closes_physical_graph_without_kinematic_labels():
    mujoco = pytest.importorskip("mujoco")
    from exploration.closed_loop_probe import closed_physical_edges
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
      <geom size=".1"/><body name="a"><joint name="a"/><geom size=".1"/>
        <body name="rotor"><joint name="c"/><geom size=".1"/></body></body>
      <body name="b"><joint name="b"/><geom size=".1"/></body>
      </body></worldbody><equality><weld body1="rotor" body2="b"/></equality></mujoco>''')
    ids = [model.joint(name).id for name in ("a", "b", "c")]
    assert set(closed_physical_edges(model, ids)) == {(0, 1), (0, 2), (1, 2)}


def test_unexcited_data_does_not_create_motion_factors():
    x = np.ones((10, 8))
    result = response_relations(x, x * .5, x * .3)
    assert result["status"] == "insufficient_command_rank"
    assert not result["pairs"] and not result["factors"]


def test_general_many_joint_relation_and_heldout_contradiction():
    rng = np.random.default_rng(2)
    matrix = np.eye(5)
    matrix[4] = matrix[0] + matrix[1] - matrix[2]
    x = rng.normal(size=(50, 5))
    y = x @ matrix.T
    check = rng.normal(size=(20, 5)) @ matrix.T
    result = response_relations(x, y, check)
    assert result["observed_rank"] == 4
    assert len(result["factors"]) == 1
    assert result["factors"][0]["heldout_rms_deg"] < 1e-10
    contradicted = response_relations(x, y, rng.normal(size=(20, 5)))
    assert contradicted["factors"][0]["heldout_rms_deg"] > 1


def test_frontier_scores_use_the_same_rule_with_or_without_graph_prior():
    x = np.eye(3)
    y = np.diag([1., .1, .1])
    model = fit_response(x, y, "frontier", [], .03)
    pool = np.array([[.03, 0, 0], [0, .03, 0]])
    for method in ("frontier", "graph_frontier"):
        selected = choose_command(method, pool, x, model, np.random.default_rng(0),
                                  q=np.zeros(3), visited=np.zeros((1, 3)))
        assert selected == 0
