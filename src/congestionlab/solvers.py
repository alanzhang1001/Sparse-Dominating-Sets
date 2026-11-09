from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix


def solve_dom_mac(coverage: csr_matrix, costs: np.ndarray, mode: str, time_limit: float) -> dict:
    """Minimum total coverage cost: min sum(cost[u] x[u]) s.t. each vertex covered."""
    n = coverage.shape[0]
    # HiGHS expects a numeric sparse matrix; coverage itself is stored as bool.
    constraints_matrix = coverage.astype(np.float64).T.tocsr()
    started = time.perf_counter()
    if mode == "lp":
        result = linprog(
            costs.astype(float),
            A_ub=-constraints_matrix,
            b_ub=-np.ones(n),
            bounds=(0.0, 1.0),
            method="highs",
            # SciPy forwards the unlisted HiGHS option while warning that it is
            # solver-specific; this keeps formal solver comparisons single-threaded.
            options={"time_limit": float(time_limit), "threads": 1},
        )
        elapsed = time.perf_counter() - started
        return {
            "solver": "scipy-highs-lp",
            "status_code": int(result.status),
            "status": str(result.message),
            "seconds": elapsed,
            "objective_total": float(result.fun) if result.fun is not None else None,
            "optimality_proven": bool(result.status == 0),
            "primal_feasible": bool(result.x is not None),
            "dual_bound_total": float(result.fun) if result.status == 0 else None,
        }
    if mode == "ilp":
        result = milp(
            c=costs.astype(float),
            integrality=np.ones(n, dtype=np.uint8),
            bounds=Bounds(np.zeros(n), np.ones(n)),
            constraints=LinearConstraint(constraints_matrix, np.ones(n), np.full(n, np.inf)),
            options={"time_limit": float(time_limit), "mip_rel_gap": 0.0, "threads": 1},
        )
        elapsed = time.perf_counter() - started
        return {
            "solver": "scipy-highs-milp",
            "status_code": int(result.status),
            "status": str(result.message),
            "seconds": elapsed,
            "objective_total": float(result.fun) if result.fun is not None else None,
            "optimality_proven": bool(result.status == 0),
            "primal_feasible": bool(result.x is not None),
            "dual_bound_total": float(result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None else None,
            "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
            "node_count": int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
        }
    raise ValueError("mode must be lp or ilp")
