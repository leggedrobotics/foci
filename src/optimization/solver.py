from typing import Any

import casadi as cas


def default_ipopt_options() -> dict[str, Any]:
    """
    Default IPOPT options for trajectory optimization.
    """
    return {
        "ipopt.print_level": 3,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-3,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_obj_change_tol": 1e-3,
        "ipopt.constr_viol_tol": 1e-3,
        "ipopt.acceptable_iter": 1,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
        "print_time": 1,
    }


def create_solver(
    nlp: dict,
    name: str = "trajectory_solver",
    options: dict[str, Any] | None = None,
):
    """
    Create a CasADi NLP solver.
    """
    solver_options = default_ipopt_options()

    if options is not None:
        solver_options.update(options)

    return cas.nlpsol(
        name,
        "ipopt",
        nlp,
        solver_options,
    )