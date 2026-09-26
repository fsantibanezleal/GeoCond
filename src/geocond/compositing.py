"""Conservative continuous-interval overlap, coverage and category proportions."""

from dataclasses import dataclass

import numpy as np

from .validation import ValidationError, array, positive


@dataclass(frozen=True)
class Composite:
    start: float
    end: float
    mean: float
    numerator: float
    valid_length: float
    missing_length: float
    coverage: float
    domain: str | None
    parents: tuple[tuple[str, float], ...]
    status: str


def fixed_boundaries(start: float, end: float, length: float, *, residual: str = "keep"):
    positive(length, "composite length")
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValidationError("Composite range must be a positive measured-depth interval")
    if residual not in {"keep", "drop"}:
        raise ValidationError("Residual policy must be keep or drop")
    count = int(np.floor((end - start) / length + 1e-12))
    boundaries = start + np.arange(count + 1) * length
    if residual == "keep" and end - boundaries[-1] > 1e-12 * max(1, end):
        boundaries = np.r_[boundaries, end]
    elif np.isclose(boundaries[-1], end, atol=1e-12, rtol=1e-12):
        boundaries[-1] = end
    return boundaries


def _intervals(from_depth, to_depth, boundaries, source_ids, support_kind):
    if support_kind != "continuous-interval":
        raise ValidationError("Length compositing requires known continuous interval support")
    a, b = array(from_depth, "from_depth", ndim=1), array(to_depth, "to_depth", ndim=1)
    edges = array(boundaries, "boundaries", ndim=1)
    if len(a) != len(b) or np.any(a < 0) or np.any(b <= a):
        raise ValidationError("Every interval must have nonnegative start and positive length")
    if len(edges) < 2 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValidationError("Composite boundaries must be strictly increasing")
    order = np.argsort(a, kind="stable")
    if np.any(a[order][1:] < b[order][:-1]):
        raise ValidationError(
            "Overlapping intervals require a resolved observation policy before compositing"
        )
    ids = tuple(str(i) for i in range(len(a))) if source_ids is None else tuple(source_ids)
    if len(ids) != len(a) or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValidationError("source_ids must be unique strings matching the input rows")
    return a, b, edges, ids


def composite_intervals(
    from_depth,
    to_depth,
    values,
    boundaries,
    *,
    support_kind="continuous-interval",
    domains=None,
    source_ids=None,
    min_coverage=0,
):
    a, b, edges, ids = _intervals(from_depth, to_depth, boundaries, source_ids, support_kind)
    values = array(values, "values", ndim=1, finite=False)
    if len(values) != len(a) or np.isinf(values).any():
        raise ValidationError("Interval values must match rows and be finite or NaN for missing")
    if not np.isfinite(min_coverage) or not 0 <= min_coverage <= 1:
        raise ValidationError("min_coverage must be between zero and one")
    if domains is None:
        labels = [None] * len(a)
    else:
        labels = list(domains)
        if len(labels) != len(a) or any(x is not None and not isinstance(x, str) for x in labels):
            raise ValidationError("domains must contain one string or null per interval")
        cuts = []
        order = np.argsort(a, kind="stable")
        for i, j in zip(order[:-1], order[1:], strict=True):
            if labels[i] != labels[j]:
                cuts.extend((b[i], a[j]))
        edges = np.unique(np.r_[edges, [x for x in cuts if edges[0] < x < edges[-1]]])
    result = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        overlap = np.maximum(0, np.minimum(b, hi) - np.maximum(a, lo))
        selected = np.flatnonzero(overlap > 0)
        valid = selected[np.isfinite(values[selected])]
        length = float(overlap[valid].sum())
        numerator = float(overlap[valid] @ values[valid])
        coverage = length / (hi - lo)
        domain_set = {labels[i] for i in selected}
        if len(domain_set) > 1:
            raise ValidationError("Composite crosses unresolved domains")
        status = (
            "estimated"
            if length > 0 and coverage + 1e-14 >= min_coverage
            else "insufficient-coverage"
        )
        result.append(
            Composite(
                float(lo),
                float(hi),
                numerator / length if status == "estimated" else np.nan,
                numerator,
                length,
                float(hi - lo - length),
                float(coverage),
                next(iter(domain_set), None),
                tuple((ids[i], float(overlap[i])) for i in selected),
                status,
            )
        )
    return tuple(result)


def composite_categories(
    from_depth,
    to_depth,
    categories,
    boundaries,
    *,
    support_kind="continuous-interval",
    source_ids=None,
):
    a, b, edges, ids = _intervals(from_depth, to_depth, boundaries, source_ids, support_kind)
    labels = list(categories)
    if len(labels) != len(a) or any(
        x is not None and not isinstance(x, (str, int)) for x in labels
    ):
        raise ValidationError("Categories must be explicit string/integer codes or null")
    result = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        overlap = np.maximum(0, np.minimum(b, hi) - np.maximum(a, lo))
        lengths = {}
        for i in np.flatnonzero(overlap > 0):
            if labels[i] is not None:
                lengths[labels[i]] = lengths.get(labels[i], 0.0) + float(overlap[i])
        total = sum(lengths.values())
        result.append(
            {
                "start": float(lo),
                "end": float(hi),
                "coverage": total / (hi - lo),
                "proportions": {k: v / total for k, v in lengths.items()},
                "missing_length": float(hi - lo - total),
                "parents": tuple((ids[i], float(overlap[i])) for i in np.flatnonzero(overlap > 0)),
            }
        )
    return tuple(result)
