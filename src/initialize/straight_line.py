from __future__ import annotations

from time import perf_counter

import numpy as np

from src.initialize.initializer import InitializerResult, PathInitializer


class StraightLineInitializer(PathInitializer):
    """
    Inicializador minimo: interpolacion lineal en espacio articular entre
    start y la configuracion articular que alcanza goal (task-space),
    sin ningun tipo de planificacion de por medio.

    A diferencia de RRTStarInitializer, no comprueba colisiones ni evita
    obstaculos -- traza directamente la linea recta mas corta en espacio
    de articulaciones entre las dos configuraciones. Sirve como punto de
    partida "neutro" para los optimizadores (STOMP, CHOMP, FOCI), sin
    sesgar de antemano la topologia de la trayectoria hacia ningun lado
    del obstaculo (a diferencia de RRT*, cuya salida cruda ya puede traer
    un rodeo que los optimizadores locales solo pueden suavizar, no
    deshacer).

    Como start ya viene en espacio articular pero goal es una posicion en
    espacio de tarea (task-space), primero se resuelve una cinematica
    inversa numerica (Gauss-Newton amortiguado / damped least squares)
    usando robot.f_task(q), reutilizando la MISMA cinematica directa que
    usa el resto del pipeline (no una reimplementacion aparte).
    """

    def __init__(
        self,
        ik_iters: int = 200,
        ik_damping: float = 0.05,
        ik_step: float = 1.0,
        ik_tol: float = 1e-4,
        fd_eps: float = 1e-6,
    ):
        self.ik_iters = ik_iters
        self.ik_damping = ik_damping
        self.ik_step = ik_step
        self.ik_tol = ik_tol
        self.fd_eps = fd_eps

    def initialize(
        self,
        robot,
        environment,
        start: np.ndarray,
        goal: np.ndarray,
        num_control_points: int,
    ) -> InitializerResult:
        t0 = perf_counter()

        start = np.asarray(start, dtype=float)
        goal_task = np.asarray(goal, dtype=float).reshape(-1)

        goal_joint, ik_error = self._numeric_ik(robot, goal_task, start)

        s = np.linspace(0.0, 1.0, num_control_points)[:, None]
        control_points = (1.0 - s) * start[None, :] + s * goal_joint[None, :]

        t1 = perf_counter()

        return InitializerResult(
            control_points=control_points,
            trajectory=control_points,
            success=ik_error < 1e-2,
            timings={"initializer": t1 - t0},
            metadata={
                "type": "straight_line",
                "goal_joint": goal_joint,
                "ik_error": ik_error,
            },
        )

    # ------------------------------------------------------------------
    # Cinematica inversa numerica (damped least squares), reutilizando
    # robot.f_task para no duplicar la cinematica directa del robot.
    # ------------------------------------------------------------------

    def _f_task_np(self, robot, q: np.ndarray) -> np.ndarray:
        point = robot.f_task(q)

        if hasattr(point, "full"):
            point = point.full()

        return np.asarray(point, dtype=float).reshape(3)

    def _jacobian_fd(self, robot, q: np.ndarray) -> np.ndarray:
        n = q.shape[0]
        J = np.zeros((3, n))
        f0 = self._f_task_np(robot, q)

        for joint_idx in range(n):
            q_perturbed = q.copy()
            q_perturbed[joint_idx] += self.fd_eps
            J[:, joint_idx] = (self._f_task_np(robot, q_perturbed) - f0) / self.fd_eps

        return J

    def _numeric_ik(
        self, robot, target: np.ndarray, q0: np.ndarray
    ) -> tuple[np.ndarray, float]:
        q = q0.copy()

        for _ in range(self.ik_iters):
            pos = self._f_task_np(robot, q)
            err = target - pos

            if np.linalg.norm(err) < self.ik_tol:
                break

            J = self._jacobian_fd(robot, q)
            JJt = J @ J.T + self.ik_damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = q + self.ik_step * dq

        final_error = float(np.linalg.norm(target - self._f_task_np(robot, q)))

        return q, final_error
