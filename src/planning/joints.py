from dataclasses import dataclass, field

@dataclass(frozen=True)
class JointGroups:
    """
    Defines real and virtual joints for optimization.

    Real joints:
        subject to wmax / amax.

    Virtual joints:
        subject to virtual_wmax / virtual_amax.

    Jerk cost is computed separately for both groups.
    """

    virtual_indices: list[int] = field(default_factory=list)
    virtual_wmax: float = 1.0
    virtual_amax: float = 1.0
    real_wmax: float = 1.0
    real_amax: float = 1.0

    def real_indices(self, n_dof: int) -> list[int]:
        virtual = set(self.virtual_indices)
        return [idx for idx in range(n_dof) if idx not in virtual]

    def validate(self, n_dof: int) -> None:
        for idx in self.virtual_indices:
            if idx < 0 or idx >= n_dof:
                raise ValueError(
                    f"Virtual joint index {idx} is invalid for n_dof={n_dof}."
                )