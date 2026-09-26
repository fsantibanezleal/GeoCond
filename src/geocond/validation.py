"""Strict numerical validation and explicit failure types."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ValidationError(ValueError):
    """Shared input violates a scientific or numerical contract."""


class NumericalError(RuntimeError):
    """A scientifically specified system cannot be solved within its policy."""


class CancelledError(RuntimeError):
    """Computation was cancelled; no complete result is available."""


def array(
    value: ArrayLike, name: str, *, ndim: int | None = None, finite: bool = True
) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValidationError(f"{name} must contain numeric values, not strings or booleans")
    result = np.array(raw, dtype=np.float64, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValidationError(f"{name} must have {ndim} dimensions")
    if finite and not np.isfinite(result).all():
        raise ValidationError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValidationError(f"{name} must be a finite number")
    if value < 0 if zero else value <= 0:
        raise ValidationError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def cancel_if_requested(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise CancelledError("Computation cancelled before a complete result was produced")


def rotation_matrix(value: ArrayLike | None) -> NDArray[np.float64]:
    result = array(np.eye(3) if value is None else value, "rotation", ndim=2)
    if result.shape != (3, 3) or not np.allclose(result.T @ result, np.eye(3), atol=1e-10):
        raise ValidationError("rotation must be an orthonormal 3 by 3 matrix")
    if not np.isclose(np.linalg.det(result), 1, atol=1e-10):
        raise ValidationError("rotation must be proper and right handed")
    return result
