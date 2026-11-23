"""Small general-dimensional active search heuristics, without mechanical labels.

The state graph contains executed local motions, not presumed motion stages.
Graph methods use known assembly distance as a soft response-model prior.
"""
import numpy as np

from .active_motion import ActiveMotionLearner
from .graph_response import ResponseModel, graph_penalty

METHODS = ("random", "information", "graph_information", "frontier", "graph_frontier")


def candidate_commands(lower, upper, amplitude, rng, *, matrix=None):
    n = len(lower)
    candidates, kinds, seen = [], [], set()

    def add(direction, kind):
        value = ActiveMotionLearner.bounded_direction(direction, lower, upper, amplitude)
        if value is not None:
            key = tuple(np.round(value, 10))
            if key not in seen:
                seen.add(key)
                candidates.append(value)
                kinds.append(kind)

    for i in range(n):
        for sign in (-1, 1):
            add(np.eye(n)[i] * sign, "single")
    # All pairs, including non-neighbours; no fixed pairings or group count.
    for i in range(n):
        for j in range(i + 1, n):
            for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                direction = np.zeros(n)
                direction[[i, j]] = [a, b]
                add(direction, "pair")
    for _ in range(64):
        direction = rng.normal(size=n)
        direction = np.where(upper <= 1e-10, -abs(direction), direction)
        direction = np.where(lower >= -1e-10, abs(direction), direction)
        add(direction, "multi")
    if matrix is not None:
        _, singular, vt = np.linalg.svd(matrix)
        active = singular > max(.06, .12 * singular[0])
        # Right singular vectors live in COMMAND space, unlike output factors.
        basis = vt[active].T
        for direction in basis.T:
            add(direction, "learned_command_direction")
            add(-direction, "learned_command_direction")
        for _ in range(24):
            if basis.shape[1]:
                add(basis @ rng.normal(size=basis.shape[1]), "learned_command_combination")
    return np.asarray(candidates), kinds


def fit_response(commands, responses, method, edges, amplitude):
    n = np.asarray(commands).shape[1]
    penalty = graph_penalty(n, edges) if method.startswith("graph_") else np.ones((n, n))
    return ResponseModel.fit(commands, responses, penalty, regularization=.003, scale=amplitude)


def uncertainty_scores(candidates, commands, model):
    x = np.asarray(commands) / model.scale
    c = candidates / model.scale
    covariance = x.T @ x / len(x)
    inverses = np.linalg.inv(covariance[None] + model.regularization *
                            np.array([np.diag(row) for row in model.penalty]))
    return np.log1p(np.einsum("ki,ij,kj->k", c, inverses.mean(axis=0), c))


def choose_command(method, candidates, commands, model, rng, *, q=None, visited=None):
    if method == "random":
        return int(rng.integers(len(candidates)))
    uncertainty = uncertainty_scores(candidates, commands, model)
    if not method.endswith("frontier") or q is None:
        return int(np.argmax(uncertainty))
    predicted = model.predict(candidates)
    endpoints = q + predicted
    novelty = np.min(np.linalg.norm(endpoints[:, None] - visited[None], axis=2), axis=1)
    # Novelty is measured in observed joint space, not an assumed 2-D progress chart.
    score = novelty / max(model.scale, 1e-12) + .15 * uncertainty / max(uncertainty.max(), 1e-12)
    return int(np.argmax(score))


def response_relations(commands, responses, check_responses, *, threshold=.12):
    """Data-supported local factors, with held-out residuals and weak rows excluded."""
    X, Y, check = map(np.asarray, (commands, responses, check_responses))
    n = X.shape[1]
    if np.linalg.matrix_rank(X) < n:
        return {"status": "insufficient_command_rank", "pairs": [], "factors": []}
    _, singular, vt = np.linalg.svd(Y, full_matrices=False)
    rank = int(sum(singular > max(singular[0] * threshold, 1e-6)))
    pairs = []
    rms = np.sqrt(np.mean(Y ** 2, axis=0))
    for i in range(n):
        for j in range(i + 1, n):
            if min(rms[i], rms[j]) < .08 * max(rms):
                continue
            slope = float(Y[:, i] @ Y[:, j] / (Y[:, i] @ Y[:, i]))
            relative = np.linalg.norm(check[:, j] - slope * check[:, i]) / max(
                np.linalg.norm(check[:, j]), 1e-9)
            if relative < .12:
                pairs.append({"members": [i, j], "slope": slope, "heldout_relative_error": float(relative),
                              "status": "local_response_relation_not_global_mechanical_rule"})
    factors = [{"normal": normal.tolist(),
                "heldout_rms_deg": float(np.rad2deg(np.sqrt(np.mean((check @ normal) ** 2))))}
               for normal in vt[rank:]]
    return {"status": "finite_horizon_response_only", "observed_rank": rank,
            "singular_values": singular.tolist(), "pairs": pairs, "factors": factors}
