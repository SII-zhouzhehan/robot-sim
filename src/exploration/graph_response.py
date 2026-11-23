"""Conservative graph-prior response identification, not a GNN or RL policy.

Physical graph distances change shrinkage strengths, never impose zero edges or
angle equalities. Response factors live in measured motion space; coordinated
commands must be solved through the identified actuator response matrix.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


def graph_penalty(count, edges):
    """Soft distance prior: distant effects are allowed, with stronger shrinkage."""
    if count < 1:
        raise ValueError("Expected at least one joint")
    distance = np.full((count, count), np.inf)
    np.fill_diagonal(distance, 0)
    for left, right in edges:
        if left == right or not (0 <= left < count and 0 <= right < count):
            raise ValueError("Invalid physical graph edge")
        distance[left, right] = distance[right, left] = 1
    for pivot in range(count):
        distance = np.minimum(distance, distance[:, pivot, None] + distance[None, pivot, :])
    return 1 + np.minimum(distance, 3) ** 2


@dataclass
class ResponseModel:
    matrix: np.ndarray
    penalty: np.ndarray
    regularization: float
    scale: float
    command_rank: int
    diagonal: bool = False

    @classmethod
    def fit(cls, commands, responses, penalty, *, regularization=.001, scale=.02,
            diagonal=False):
        X, Y, W = (np.asarray(value, dtype=float) for value in (commands, responses, penalty))
        if (X.ndim != 2 or Y.shape != X.shape or len(X) == 0
                or W.shape != (X.shape[1], X.shape[1])
                or not all(np.isfinite(value).all() for value in (X, Y, W))
                or np.any(W <= 0) or not np.isfinite([regularization, scale]).all()
                or regularization <= 0 or scale <= 0):
            raise ValueError("Invalid response data or regularizer")
        xn, yn = X / scale, Y / scale
        covariance = xn.T @ xn / len(X)
        cross = xn.T @ yn / len(X)
        matrix = np.zeros_like(W)
        for target in range(X.shape[1]):
            if diagonal:
                matrix[target, target] = cross[target, target] / (
                    covariance[target, target] + regularization * W[target, target])
            else:
                matrix[target] = np.linalg.solve(
                    covariance + regularization * np.diag(W[target]), cross[:, target])
        return cls(matrix, W.copy(), float(regularization), float(scale),
                   int(np.linalg.matrix_rank(X)), diagonal)

    def predict(self, commands):
        commands = np.asarray(commands, dtype=float)
        if commands.shape[-1] != len(self.matrix) or not np.isfinite(commands).all():
            raise ValueError("Invalid command shape or values")
        return commands @ self.matrix.T

    def command_for_motion(self, desired_motion, lower, upper, *, amplitude=.02, damping=.01):
        """Box-constrained response inversion; motion factors are NOT command rules."""
        n = len(self.matrix)
        desired, lower, upper = (np.asarray(value, dtype=float)
                                 for value in (desired_motion, lower, upper))
        if (any(value.shape != (n,) or not np.isfinite(value).all()
                for value in (desired, lower, upper))
                or np.any(lower > 0) or np.any(upper < 0) or np.any(lower > upper)
                or not np.isfinite([amplitude, damping]).all() or amplitude <= 0 or damping <= 0):
            raise ValueError("Invalid motion request or command bounds")
        free = upper - lower > 1e-12
        command = np.zeros(n)
        if not free.any():
            return command
        matrix = np.vstack((self.matrix[:, free], np.sqrt(damping) * np.eye(n)[:, free]))
        target = np.concatenate((desired, np.zeros(n)))
        solution = lsq_linear(matrix, target, bounds=(lower[free], upper[free]), tol=1e-10)
        if not solution.success:
            raise RuntimeError("Local response inversion did not converge")
        command[free] = solution.x
        norm = np.linalg.norm(command)
        if norm > amplitude:
            # Zero lies inside the box; uniform scaling keeps every bound valid.
            command *= amplitude / norm
        return command


def select_response_model(commands, responses, validation_commands, validation_responses,
                          penalty, *, scale=.02, diagonal=False,
                          grid=(.0001, .001, .01, .1)):
    """Regularization uses a separate validation set, never the final test set."""
    choices = []
    for regularization in grid:
        model = ResponseModel.fit(commands, responses, penalty,
                                  regularization=regularization, scale=scale, diagonal=diagonal)
        error = float(np.sqrt(np.mean((model.predict(validation_commands) - validation_responses) ** 2)))
        choices.append((error, model))
    error, model = min(choices, key=lambda choice: choice[0])
    return model, error


def response_graph_evidence(model, commands, responses, check_responses, *, names=None,
                            bootstrap=80, seed=0, effect_threshold=.05, rank_threshold=.08,
                            relation_tolerance=.1):
    """Report repeatable influences and candidate output relations with evidence.

    This is exploratory evidence for one control/pose setup, not a mechanical
    constraint certificate. Bootstrap intervals omit simulator/model bias.
    Unresponsive or underexcited joints are never labeled permanently locked.
    """
    X, Y, check = (np.asarray(value, dtype=float) for value in
                   (commands, responses, check_responses))
    n = len(model.matrix)
    if (X.ndim != 2 or Y.shape != X.shape or X.shape[1] != n or len(X) < 1
            or check.ndim != 2 or check.shape[1] != n or len(check) < 1
            or not all(np.isfinite(value).all() for value in (X, Y, check))
            or bootstrap < 1 or effect_threshold <= 0 or rank_threshold <= 0
            or relation_tolerance <= 0):
        raise ValueError("Invalid graph evidence inputs")
    names = list(names) if names is not None else [f"joint{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError("Expected one name per joint")
    input_values = np.linalg.svd(X / model.scale, compute_uv=False)
    condition = float(input_values[-1] / input_values[0]) if input_values[0] > 0 else 0.
    sufficiently_excited = model.command_rank == n and condition >= .05
    response_values = np.linalg.svd(model.matrix, compute_uv=False)
    rank = int(np.count_nonzero(response_values > rank_threshold))
    result = {"nodes": [{"id": i, "name": name} for i, name in enumerate(names)],
              "input_rank": model.command_rank, "input_condition_ratio": condition,
              "sufficiently_excited": sufficiently_excited,
              "observed_response_rank": rank,
              "rank_threshold_sensitivity": {str(t): int(np.count_nonzero(response_values > t))
                                             for t in (.04, .08, .16)},
              "response_singular_values": response_values.tolist(),
              "bootstrap_full_input_rank_count": 0, "bootstrap_count": bootstrap,
              "response_edges": [], "candidate_motion_factors": [],
              "weak_response_directions": [],
              "interpretation": "Finite-horizon servo response; no certified mechanical constraints"}
    if not sufficiently_excited:
        result["status"] = "insufficient_input_excitation"
        return result
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        indices = rng.integers(len(X), size=len(X))
        if np.linalg.matrix_rank(X[indices]) != n:
            continue
        fitted = ResponseModel.fit(X[indices], Y[indices], model.penalty,
                                   regularization=model.regularization, scale=model.scale,
                                   diagonal=model.diagonal)
        samples.append(fitted.matrix)
    result["bootstrap_full_input_rank_count"] = len(samples)
    if len(samples) < bootstrap * .8:
        result["status"] = "insufficient_bootstrap_excitation"
        return result
    samples = np.asarray(samples)
    lo, hi = np.quantile(samples, [.025, .975], axis=0)
    for target in range(n):
        for source in range(n):
            if source != target and (lo[target, source] > effect_threshold
                                     or hi[target, source] < -effect_threshold):
                result["response_edges"].append({"source": source, "target": target,
                    "weight": float(model.matrix[target, source]),
                    "bootstrap_interval": [float(lo[target, source]), float(hi[target, source])],
                    "status": "repeatable_response_candidate_not_mechanical_coupling"})
    _, _, right = np.linalg.svd(model.matrix.T)
    sample_ranks = np.sum(np.linalg.svd(samples, compute_uv=False) > rank_threshold, axis=1)
    stability = float(np.mean(sample_ranks == rank))
    result["response_rank_bootstrap_agreement"] = stability
    for normal in right[rank:]:
        residual = float(np.sqrt(np.mean((check @ normal) ** 2)))
        support = np.flatnonzero(abs(normal) > 1e-12)
        dominant = np.flatnonzero(abs(normal) >= .1)
        factor = {"members": support.tolist(), "coefficients": normal[support].tolist(),
                  "full_normal": normal.tolist(), "dominant_members": dominant.tolist(),
                  "check_rms_rad": residual, "rank_stability": stability}
        if len(dominant) < 2:
            factor["status"] = "weak_response_not_a_locked_joint"
            result["weak_response_directions"].append(factor)
            continue
        signal = float(np.sqrt(np.mean(np.sum((check[:, dominant] * normal[dominant]) ** 2, axis=1))))
        relative = residual / max(signal, model.scale * 1e-3)
        factor.update(relative_check_residual=relative,
                      status="validated_local_response_relation" if stability >= .8 and
                      relative <= relation_tolerance and len(X) >= 2 * n else "unconfirmed_local_relation")
        result["candidate_motion_factors"].append(factor)
    result["status"] = "local_response_evidence_only"
    return result
