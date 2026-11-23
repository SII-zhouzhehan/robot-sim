"""A small active-identification prototype; no rewards, policy training or Phi.

The model maps a bounded command increment to the observed joint displacement.
It estimates a local closed-loop response, NOT the robot's exact geometric
constraint Jacobian. Joint velocity feedback can be integrated by the adapter.
Physical adjacency supplies probe candidates; learned factors supply coordinated
probes and motion projection. Neither adjacency nor correlation is a hard rule.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeCandidate:
    increment: np.ndarray
    kind: str
    members: tuple[int, ...]


class ActiveMotionLearner:
    def __init__(self, joint_names, physical_edges=(), *, ridge=1e-6,
                 rank_threshold=.08, coupling_tolerance=.08, seed=0):
        self.names = tuple(joint_names)
        self.n = len(self.names)
        if self.n == 0 or len(set(self.names)) != self.n:
            raise ValueError("Expected unique, nonempty joint names")
        if ridge <= 0 or rank_threshold <= 0 or not 0 < coupling_tolerance < 1:
            raise ValueError("Invalid identification thresholds")
        self.edges = tuple(sorted(set(tuple(sorted(map(int, e))) for e in physical_edges)))
        if any(len(e) != 2 or e[0] == e[1] or min(e) < 0 or max(e) >= self.n for e in self.edges):
            raise ValueError("Physical edges must connect two different joint indices")
        self.ridge = ridge
        self.rank_threshold = rank_threshold
        self.coupling_tolerance = coupling_tolerance
        self.rng = np.random.default_rng(seed)
        self.commands, self.responses = [], []
        self.matrix = np.zeros((self.n, self.n))
        self.information = np.eye(self.n) * ridge
        self.command_rank = 0
        self.observed_rank = None
        self.basis = np.empty((self.n, 0))
        self.constraints = np.empty((0, self.n))
        self.singular_values = np.zeros(self.n)
        self.factors = []
        self.kind_counts = {}

    def _vector(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.n,) or not np.isfinite(value).all():
            raise ValueError(f"Expected {self.n} finite values")
        return value

    def observe(self, increment, displacement, *, kind="external"):
        x, y = self._vector(increment), self._vector(displacement)
        self.commands.append(x.copy())
        self.responses.append(y.copy())
        self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1
        X, Y = np.asarray(self.commands), np.asarray(self.responses)
        self.information = X.T @ X + self.ridge * np.eye(self.n)
        self.matrix = np.linalg.solve(self.information, X.T @ Y).T
        self.command_rank = int(np.linalg.matrix_rank(X))
        # Underexcitation can create perfectly correlated rows of a ridge fit.
        # Do not call those correlations constraints before probing a full span.
        self.factors = []
        if self.command_rank < self.n:
            self.observed_rank = None
            self.basis = np.empty((self.n, 0))
            self.constraints = np.empty((0, self.n))
            return
        U, values, _ = np.linalg.svd(self.matrix, full_matrices=True)
        self.singular_values = values
        rank = int(np.count_nonzero(values > self.rank_threshold))
        self.observed_rank = rank
        self.basis = U[:, :rank]
        self.constraints = U[:, rank:].T
        self.factors = self._coupled_groups()

    def _coupled_groups(self):
        """Proportional response rows imply a candidate local motion ratio.

        This works for any group size and does not require physical adjacency.
        General multi-joint linear relations remain in ``constraints`` even
        when they cannot be expressed as groups of proportional rows.
        """
        norms = np.linalg.norm(self.matrix, axis=1)
        remaining = {i for i in range(self.n) if norms[i] > self.rank_threshold}
        result = []
        while remaining:
            anchor = max(remaining, key=lambda i: (norms[i], -i))
            reference = self.matrix[anchor]
            group, coefficients = [anchor], [1.0]
            for i in sorted(remaining - {anchor}):
                slope = float(self.matrix[i] @ reference / (reference @ reference))
                error = np.linalg.norm(self.matrix[i] - slope * reference) / norms[i]
                if error <= self.coupling_tolerance:
                    # Avoid chaining several approximate matches into a false group.
                    if all(np.linalg.norm(self.matrix[i] / slope - self.matrix[j] / c)
                           <= self.coupling_tolerance * np.linalg.norm(reference)
                           for j, c in zip(group, coefficients)):
                        group.append(i)
                        coefficients.append(slope)
            remaining.difference_update(group)
            if len(group) > 1:
                order = np.argsort(group)
                result.append({"members": [group[i] for i in order],
                               "coefficients": [coefficients[i] for i in order],
                               "status": "candidate_local_relation"})
        return result

    @staticmethod
    def bounded_direction(direction, lower, upper, amplitude):
        """Scale a whole direction to bounds; component clipping breaks coupling."""
        d = np.asarray(direction, dtype=float).copy()
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return None
        d *= amplitude / norm
        scale = 1.0
        for value, lo, hi in zip(d, lower, upper):
            if value > 1e-12:
                scale = min(scale, hi / value)
            elif value < -1e-12:
                scale = min(scale, lo / value)
        if scale <= 1e-6:
            return None
        return d * scale

    def candidates(self, lower, upper, amplitude):
        lower, upper = self._vector(lower), self._vector(upper)
        if np.any(lower > 0) or np.any(upper < 0) or amplitude <= 0:
            raise ValueError("Increment bounds must contain zero; amplitude must be positive")
        candidates, seen = [], set()

        def add(direction, kind):
            d = self.bounded_direction(direction, lower, upper, amplitude)
            if d is None:
                return
            key = tuple(np.round(d, 10))
            if key not in seen:
                seen.add(key)
                candidates.append(ProbeCandidate(d, kind, tuple(np.flatnonzero(abs(d) > 1e-9))))

        for i in range(self.n):
            for sign in (-1, 1):
                add(np.eye(self.n)[i] * sign, "single_joint")
        # First identify individual causal responses instead of fitting correlated
        # multi-joint trajectories generated by a known formula.
        if self.command_rank < self.n:
            return candidates
        for left, right in self.edges:
            for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                d = np.zeros(self.n)
                d[[left, right]] = signs
                add(d, "physical_graph_pair")
        for factor in self.factors:
            d = np.zeros(self.n)
            d[factor["members"]] = factor["coefficients"]
            add(d, "learned_graph_group")
            add(-d, "learned_graph_group")
        # Left-nullspace factors constrain the output motion. Project requests
        # through ALL factors, including relations involving three or more joints.
        for i in range(self.n):
            for sign in (-1, 1):
                requested = np.eye(self.n)[i] * sign
                projected = self.basis @ (self.basis.T @ requested)
                add(projected, "learned_subspace")
        # Keep nonlocal multi-joint probes available; adjacency is not a constraint.
        for _ in range(24):
            d = self.rng.normal(size=self.n)
            d = np.where(upper <= 1e-10, -abs(d), d)
            d = np.where(lower >= -1e-10, abs(d), d)
            add(d, "nonlocal_probe")
        return candidates

    def choose(self, lower, upper, amplitude, *, use_graph=True):
        options = self.candidates(lower, upper, amplitude)
        if not use_graph:
            options = [c for c in options if c.kind not in ("physical_graph_pair", "learned_graph_group")]
        if not options:
            raise RuntimeError("No nonzero probe fits the current bounds")
        inverse = np.linalg.inv(self.information)
        gains = np.array([np.log1p(c.increment @ inverse @ c.increment) for c in options])
        gains /= max(float(gains.max()), 1e-12)
        # An explicit heuristic, not a trained RL policy or a novelty claim:
        # information gain plus coverage of graph-derived validation actions.
        bonus = np.array([.2 / (1 + self.kind_counts.get(c.kind, 0))
                          if c.kind in ("physical_graph_pair", "learned_graph_group") else 0
                          for c in options])
        selected = int(np.argmax(gains + bonus))
        return options[selected], float(gains[selected] + bonus[selected])

    def predict(self, increment):
        return self.matrix @ self._vector(increment)

    def coordinated_request(self, requested, lower, upper, amplitude):
        if self.observed_rank is None:
            raise RuntimeError("Probe a full command span before reusing relations")
        d = self.basis @ (self.basis.T @ self._vector(requested))
        return self.bounded_direction(d, self._vector(lower), self._vector(upper), amplitude)

    def graph(self):
        responsive = np.linalg.norm(self.matrix, axis=1) > self.rank_threshold
        influences = []
        if self.command_rank == self.n:
            for target, row in enumerate(self.matrix):
                threshold = max(self.rank_threshold, .15 * np.max(abs(row)))
                for source in np.flatnonzero(abs(row) > threshold):
                    if source != target:
                        influences.append({"source": int(source), "target": target,
                                           "weight": float(row[source]),
                                           "kind": "finite_horizon_response_not_hard_coupling"})
        constraints = []
        for row in self.constraints:
            members = np.flatnonzero(abs(row) > 1e-3)
            constraints.append({"members": members.tolist(), "coefficients": row[members].tolist(),
                                "status": "candidate_local_linear_relation"})
        return {"nodes": [{"id": i, "name": name,
                           "status": "response_observed" if responsive[i] else "response_unresolved"}
                          for i, name in enumerate(self.names)],
                "physical_edges": [list(e) for e in self.edges],
                "response_edges": influences, "coupled_groups": self.factors,
                "linear_factors": constraints, "command_rank": self.command_rank,
                "observed_response_rank": self.observed_rank,
                "response_matrix": self.matrix.tolist(),
                "singular_values": self.singular_values.tolist(),
                "probe_kinds": self.kind_counts.copy(),
                "interpretation": "Local closed-loop response; not certified geometric constraints"}
