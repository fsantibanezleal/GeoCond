"""Direct Sampling against a plain-Python oracle of its definition, CUDA candidate identity, and TI structure."""

import math

import numpy as np
import pytest

from geocond.direct_sampling import direct_sampling
from geocond.validation import CancelledError, ValidationError


def oracle(ti, shape, kinds, hard=None, active=None, ti_mask=None, max_neighbors=24, radius=math.inf, threshold=0.0,
           scan_fraction=1.0, seed=0, distance_power=0.0, zones=None):
    """The documented definition, one candidate at a time in plain Python loops."""
    ti = np.asarray(ti, float)
    if ti.ndim == 3:
        ti = ti[..., None]
    nv = ti.shape[3]
    valid = np.all(np.isfinite(ti), axis=3) & (np.ones(ti.shape[:3], bool) if ti_mask is None else ti_mask)
    vals = ti[valid]
    inverse = [1.0 / (1.0 if k == "categorical" or np.ptp(vals[:, v]) == 0 else float(np.ptp(vals[:, v])))
               for v, k in enumerate(kinds)]
    sim = np.full(tuple(shape) + (nv,), np.nan)
    is_hard = np.zeros(shape, bool)
    if hard is not None:
        for (i, j, k), v in zip(hard[0], np.asarray(hard[1], float).reshape(len(hard[0]), -1), strict=True):
            sim[i, j, k] = v
            is_hard[i, j, k] = True
    act = np.ones(shape, bool) if active is None else active
    centers = [tuple(c) for c in np.argwhere(valid)]
    nodes = np.flatnonzero((act & ~is_hard).ravel(order="F"))
    path = nodes[np.random.default_rng([seed, 0]).permutation(len(nodes))]
    informed = [(tuple(c), int(np.ravel_multi_index(tuple(c), shape, order="F"))) for c in np.argwhere(is_hard)]
    chosen_ids = {}
    for node in path:
        x = np.unravel_index(node, shape, order="F")
        cand = sorted(((math.dist(c, x), fid, c) for c, fid in informed), key=lambda t: (t[0], t[1]))
        cand = [t for t in cand if t[0] <= radius][:max_neighbors]
        offsets = [tuple(int(a) - int(b) for a, b in zip(c, x, strict=True)) for _, _, c in cand]
        data = [sim[c] for _, _, c in cand]
        raw = [math.hypot(*d) ** (-distance_power) if distance_power else 1.0 for d in offsets]
        w = [r / sum(raw) for r in raw]
        pool = centers if zones is None else [c for c in centers if zones[0][c] == zones[1][x]]
        order = np.random.default_rng([seed, 1, int(node)]).permutation(len(pool))
        max_scan = max(1, math.ceil(scan_fraction * len(pool))) if pool else 0
        pick, best, best_score = None, None, math.inf
        for pos in order[:max_scan]:
            c = pool[pos]
            ok = True
            tv = []
            for d in offsets:
                q = tuple(ci + di for ci, di in zip(c, d, strict=True))
                if not all(0 <= q[a] < ti.shape[a] for a in range(3)) or not valid[q]:
                    ok = False
                    break
                tv.append(ti[q])
            if not ok:
                continue
            total = 0.0
            for v, kind in enumerate(kinds):
                acc = 0.0
                for i in range(len(offsets)):
                    if kind == "categorical":
                        acc = acc + w[i] * float(tv[i][v] != data[i][v])
                    else:
                        diff = (tv[i][v] - data[i][v]) * inverse[v]
                        acc = acc + w[i] * (diff * diff)
                total = total + (acc if kind == "categorical" else math.sqrt(acc))
            score = total * (1.0 / len(kinds))
            if score <= threshold:
                pick = c
                break
            if score < best_score:
                best, best_score = c, score
        if pick is None:
            if best is None:
                continue
            pick = best
        sim[x] = ti[pick]
        chosen_ids[x] = int(np.ravel_multi_index(pick, ti.shape[:3], order="F"))
        informed.append((tuple(int(a) for a in x), int(node)))
    return sim, chosen_ids


def channel_ti(nx=40, ny=40):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    field = np.sin(x / 5.0 + 1.3 * np.sin(y / 8.0))
    return (field > 0.5).astype(float)[..., None]


def tiny_ti():
    rng = np.random.default_rng(1)
    return rng.integers(0, 3, size=(7, 6, 2)).astype(float)


def assert_same_as_oracle(result, expected_sim, expected_ids):
    assert np.array_equal(result.realization, expected_sim, equal_nan=True)
    ids = {tuple(int(v) for v in np.unravel_index(i, result.candidate.shape, order="C")): result.candidate.flat[i]
           for i in np.flatnonzero(result.candidate.ravel() >= 0)}
    assert ids == expected_ids


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_numpy_engine_equals_the_plain_python_definition(seed):
    ti = tiny_ti()
    hard = (np.array([[0, 0, 0], [3, 2, 1]]), np.array([2.0, 0.0]))
    kw = {"max_neighbors": 5, "threshold": 0.2, "scan_fraction": 0.6, "seed": seed}
    out = direct_sampling(ti, (5, 4, 2), variable_kinds=["categorical"], hard_data=hard, **kw)
    sim, ids = oracle(ti, (5, 4, 2), ["categorical"], hard=hard, **kw)
    assert_same_as_oracle(out, sim, ids)


def test_continuous_and_multivariate_scores_follow_the_definition():
    rng = np.random.default_rng(3)
    ti = np.stack([rng.normal(size=(8, 7, 1)), rng.integers(0, 2, size=(8, 7, 1)).astype(float)], axis=3)
    kinds = ["continuous", "categorical"]
    kw = {"max_neighbors": 6, "threshold": 0.15, "scan_fraction": 0.8, "seed": 4, "distance_power": 1.0}
    out = direct_sampling(ti, (6, 5, 1), variable_kinds=kinds, **kw)
    sim, ids = oracle(ti, (6, 5, 1), kinds, **kw)
    assert_same_as_oracle(out, sim, ids)


@pytest.mark.parametrize("seed", [0, 3])
def test_zones_follow_the_definition(seed):
    ti = tiny_ti()
    ti_zones = np.broadcast_to(np.arange(2)[None, None, :], ti.shape[:3]).astype(int)  # one zone per layer
    grid_zones = np.broadcast_to(np.arange(2)[None, None, :], (5, 4, 2)).astype(int)
    hard = (np.array([[0, 0, 0], [3, 2, 1]]), np.array([2.0, 0.0]))
    kw = {"max_neighbors": 5, "threshold": 0.2, "scan_fraction": 0.6, "seed": seed}
    out = direct_sampling(ti, (5, 4, 2), variable_kinds=["categorical"], hard_data=hard, zones=(ti_zones, grid_zones),
                          **kw)
    sim, ids = oracle(ti, (5, 4, 2), ["categorical"], hard=hard, zones=(ti_zones, grid_zones), **kw)
    assert_same_as_oracle(out, sim, ids)
    assert out.parameters["zones"] == {"0": 42, "1": 42}


def test_zones_confine_candidates_and_one_zone_is_the_unzoned_engine():
    ti = channel_ti(40, 40)
    shape = (24, 20, 1)
    kw = {"variable_kinds": ["categorical"], "max_neighbors": 12, "threshold": 0.1, "scan_fraction": 0.5, "seed": 7}
    plain = direct_sampling(ti, shape, **kw)
    one = direct_sampling(ti, shape, zones=(np.zeros((40, 40, 1), int), np.zeros(shape, int)), **kw)
    assert np.array_equal(plain.candidate, one.candidate) and np.array_equal(plain.realization, one.realization)
    ti_zones = (np.arange(40)[None, :, None] // 10) * np.ones((40, 1, 1), int)  # four bands along y
    grid_zones = (np.arange(20)[None, :, None] // 5) * np.ones((24, 1, 1), int)
    banded = direct_sampling(ti, shape, zones=(ti_zones, grid_zones), **kw)
    simulated = np.argwhere(banded.candidate >= 0)
    assert len(simulated) == np.prod(shape)
    for i, j, k in simulated:
        c = np.unravel_index(banded.candidate[i, j, k], ti.shape[:3], order="F")
        assert ti_zones[c] == grid_zones[i, j, k]


def test_a_layered_prior_keeps_its_order_only_with_zones():
    """Five units of four layers with sparse lenses of the unit two layers up. With few neighbours a mid-depth node is
    ambiguous, so without zones the unconditioned grid loses the order; scanning each layer's own TI layer keeps it."""
    rng = np.random.default_rng(4)
    codes = np.repeat(np.arange(5)[::-1].astype(float), 4)
    ti = codes[None, None, :] * np.ones((30, 30, 1))
    ti = np.where(rng.random(ti.shape) < 0.08, np.roll(ti, 2, axis=2), ti)
    layers = np.broadcast_to(np.arange(20)[None, None, :], ti.shape).astype(int)
    shape = (12, 12, 20)
    grid_layers = np.broadcast_to(np.arange(20)[None, None, :], shape).astype(int)
    kw = {"variable_kinds": ["categorical"], "max_neighbors": 4, "threshold": 0.0, "scan_fraction": 0.3, "seed": 1}
    zoned = direct_sampling(ti[..., None], shape, zones=(layers, grid_layers), **kw).realization[..., 0]
    free = direct_sampling(ti[..., None], shape, **kw).realization[..., 0]
    ti_agree = np.mean(ti == codes)
    assert abs(np.mean(zoned == codes) - ti_agree) < 0.02  # 0.957 against the TI's 0.959
    assert np.mean(free == codes) < 0.5  # 0.161


def test_an_empty_zone_leaves_its_nodes_failed_and_zones_are_validated():
    ti = channel_ti(20, 20)
    grid_zones = np.zeros((6, 4, 1), int)
    grid_zones[:, 2:] = 9  # no TI centre has zone 9
    out = direct_sampling(ti, (6, 4, 1), variable_kinds=["categorical"], seed=1,
                          zones=(np.zeros((20, 20, 1), int), grid_zones))
    assert out.failed[:, 2:].all() and not out.failed[:, :2].any() and not out.complete
    with pytest.raises(ValidationError, match="ti_zones must have the shape"):
        direct_sampling(ti, (6, 4, 1), variable_kinds=["categorical"], zones=(np.zeros((5, 5, 1), int), grid_zones))
    with pytest.raises(ValidationError, match="integers"):
        direct_sampling(ti, (6, 4, 1), variable_kinds=["categorical"],
                        zones=(np.zeros((20, 20, 1)), grid_zones))


@pytest.mark.cuda
@pytest.mark.parametrize("case", ["categorical", "continuous", "masked", "zoned"])
def test_cuda_selects_the_same_candidate_as_the_reference(case):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(9)
    if case == "continuous":
        ti = rng.normal(size=(20, 18, 3))
        kinds = ["continuous"]
    else:
        ti = channel_ti(30, 30)[:, :, :1]
        kinds = ["categorical"]
    ti_mask = None
    active = None
    if case == "masked":
        ti_mask = np.ones(ti.shape[:3], bool)
        ti_mask[10:14, :, :] = False
        active = np.ones((16, 12, 1), bool)
        active[:, :3, :] = False
    shape = (16, 12, 1) if case != "continuous" else (10, 8, 2)
    kw = {"variable_kinds": kinds, "ti_mask": ti_mask, "active_mask": active, "max_neighbors": 12, "threshold": 0.1,
              "scan_fraction": 0.4, "seed": 5, "candidate_chunk": 97}
    if case == "zoned":
        kw["zones"] = ((np.arange(30)[None, :, None] // 10) * np.ones((30, 1, 1), int),
                       (np.arange(12)[None, :, None] // 4) * np.ones((16, 1, 1), int))
    a = direct_sampling(ti, shape, **kw)
    b = direct_sampling(ti, shape, backend="torch", **kw)
    assert np.array_equal(a.candidate, b.candidate)
    assert np.array_equal(a.realization, b.realization, equal_nan=True)
    assert np.array_equal(a.fallback, b.fallback) and np.array_equal(a.scanned, b.scanned)
    assert np.array_equal(a.score, b.score, equal_nan=True)


def test_copied_values_follow_the_conditional_frequencies_of_the_ti():
    """One node with one informed neighbour to its east, threshold 0: the copied value must be distributed as the TI's
    values west of a cell holding the neighbour's category."""
    rng = np.random.default_rng(12)
    ti = (rng.random(size=(60, 60, 1)) < np.linspace(0.2, 0.8, 60)[None, :, None]).astype(float)
    ti[1:, :, :] = np.where(rng.random(size=(59, 60, 1)) < 0.7, ti[:-1, :, :], ti[1:, :, :])  # east-west persistence
    east = ti[1:, :, 0]
    west = ti[:-1, :, 0]
    p_expected = float(np.mean(west[east == 1.0]))
    hard = (np.array([[1, 0, 0]]), np.array([1.0]))
    active = np.zeros((2, 1, 1), bool)
    active[0, 0, 0] = True
    draws = [direct_sampling(ti, (2, 1, 1), variable_kinds=["categorical"], hard_data=hard, active_mask=active,
                             seed=s).realization[0, 0, 0, 0] for s in range(1500)]
    p_hat = float(np.mean(draws))
    assert abs(p_hat - p_expected) < 4 * math.sqrt(p_expected * (1 - p_expected) / 1500)


def test_a_score_equal_to_the_threshold_qualifies():
    ti = np.zeros((6, 6, 1))
    ti[3, :, 0] = 1.0  # a single line of category 1
    hard = (np.array([[1, 0, 0], [0, 1, 0], [2, 1, 0], [1, 2, 0]]), np.array([0.0, 0.0, 0.0, 1.0]))
    active = np.zeros((3, 3, 1), bool)
    active[1, 1, 0] = True
    # no candidate matches all four neighbours; the best mismatch exactly one of four: a score of exactly 0.25
    at = direct_sampling(ti, (3, 3, 1), variable_kinds=["categorical"], hard_data=hard, active_mask=active,
                         threshold=0.25, seed=0)
    assert at.score[1, 1, 0] == 0.25 and not at.fallback[1, 1, 0]
    below = direct_sampling(ti, (3, 3, 1), variable_kinds=["categorical"], hard_data=hard, active_mask=active,
                            threshold=0.2499, seed=0)
    assert below.fallback[1, 1, 0] and below.score[1, 1, 0] == 0.25
    assert below.scanned[1, 1, 0] == below.parameters["valid_centers"]  # the whole TI was scanned before falling back


def test_no_match_falls_back_to_the_best_scanned_candidate_and_records_it():
    ti = np.zeros((5, 5, 1))
    hard = (np.array([[0, 0, 0]]), np.array([0.0]))
    kw = {"variable_kinds": ["categorical"], "hard_data": hard, "threshold": 0.0, "seed": 1}
    ti_two = ti.copy()
    ti_two[0, 0, 0] = 1.0
    out = direct_sampling(ti_two, (2, 1, 1), **kw)
    sim, ids = oracle(ti_two, (2, 1, 1), ["categorical"], hard=hard, threshold=0.0, seed=1)
    assert_same_as_oracle(out, sim, ids)
    absent = (np.array([[0, 0, 0]]), np.array([1.0]))
    only_zero = direct_sampling(np.zeros((5, 5, 1)), (2, 1, 1), variable_kinds=["categorical"],
                                hard_data=(np.array([[0, 0, 0]]), np.array([0.0])), seed=1)
    assert only_zero.complete and not only_zero.fallback.any()
    with pytest.raises(ValidationError, match="category"):
        direct_sampling(np.zeros((5, 5, 1)), (2, 1, 1), variable_kinds=["categorical"], hard_data=absent)
    mismatch = direct_sampling(ti_two, (3, 1, 1), variable_kinds=["categorical"],
                               hard_data=(np.array([[0, 0, 0], [2, 0, 0]]), np.array([1.0, 1.0])), threshold=0.0,
                               seed=2)
    assert mismatch.fallback[1, 0, 0] and mismatch.score[1, 0, 0] > 0


def test_no_valid_candidate_leaves_the_node_failed_and_the_result_incomplete():
    ti = np.zeros((4, 4, 1))
    mask = np.zeros((4, 4, 1), bool)
    mask[0, 0, 0] = True  # one valid centre, whose east neighbour is missing
    out = direct_sampling(ti, (2, 1, 1), variable_kinds=["categorical"], ti_mask=mask,
                          hard_data=(np.array([[1, 0, 0]]), np.array([0.0])))
    assert out.failed[0, 0, 0] and not out.complete and np.isnan(out.realization[0, 0, 0, 0])


def test_one_category_domain_masks_and_hard_data():
    out = direct_sampling(np.full((5, 5, 2), 3.0), (4, 3, 2), variable_kinds=["categorical"], seed=2)
    assert np.all(out.realization == 3.0) and out.complete
    active = np.ones((6, 4, 1), bool)
    active[:2] = False
    hard = (np.array([[5, 3, 0]]), np.array([1.0]))
    out = direct_sampling(channel_ti(), (6, 4, 1), variable_kinds=["categorical"], active_mask=active, hard_data=hard)
    assert np.isnan(out.realization[:2]).all() and np.all(np.isin(out.realization[2:], [0.0, 1.0]))
    assert out.realization[5, 3, 0, 0] == 1.0 and out.hard[5, 3, 0] and out.candidate[5, 3, 0] == -1
    assert len(out.path) == active.sum() - 1
    with pytest.raises(ValidationError, match="conflicting"):
        direct_sampling(channel_ti(), (3, 3, 1), variable_kinds=["categorical"],
                        hard_data=(np.array([[1, 1, 0], [1, 1, 0]]), np.array([0.0, 1.0])))


def test_seeds_reproduce_and_differ():
    kw = {"variable_kinds": ["categorical"], "max_neighbors": 10, "threshold": 0.1, "scan_fraction": 0.5}
    a = direct_sampling(channel_ti(), (12, 12, 1), seed=4, **kw)
    b = direct_sampling(channel_ti(), (12, 12, 1), seed=4, **kw)
    c = direct_sampling(channel_ti(), (12, 12, 1), seed=5, **kw)
    assert np.array_equal(a.candidate, b.candidate) and not np.array_equal(a.candidate, c.candidate)


def stationary_channels(n=64):
    """Meandering channels with many repetitions across the image, so its edges and interior share one proportion."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (np.sin(2 * np.pi * x / 12 + 1.2 * np.sin(2 * np.pi * y / 20)) > 0.3).astype(float)[..., None]


def test_realizations_reproduce_the_ti_proportion_and_continuity():
    ti = stationary_channels()

    def lag1(field, axis):
        a = np.take(field, range(field.shape[axis] - 1), axis=axis)
        b = np.take(field, range(1, field.shape[axis]), axis=axis)
        return 0.5 * np.mean((a - b) ** 2)

    reals = [direct_sampling(ti, (32, 32, 1), variable_kinds=["categorical"], max_neighbors=20, threshold=0.05,
                             scan_fraction=0.5, seed=s).realization[..., 0, 0] for s in range(6)]
    proportion = np.mean([r.mean() for r in reals])
    assert abs(proportion - ti.mean()) < 0.06
    for axis in (0, 1):
        ratio = np.mean([lag1(r, axis) for r in reals]) / lag1(ti[..., 0], axis)
        assert 0.6 < ratio < 1.7, (axis, ratio)


def test_cancellation_leaves_no_result():
    with pytest.raises(CancelledError):
        direct_sampling(channel_ti(), (8, 8, 1), variable_kinds=["categorical"], cancel=lambda: True)
