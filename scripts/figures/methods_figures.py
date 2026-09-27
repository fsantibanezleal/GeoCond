#!/usr/bin/env python3
"""Write the method figures of docs/methods as light and dark SVG pairs (standard library only, plus GeoCond).

Usage: python scripts/figures/methods_figures.py   (from the repository root)
"""

from itertools import pairwise
from math import cos, radians, sin
from pathlib import Path

import numpy as np

from geocond.covariance import correlation

ASSETS = Path(__file__).resolve().parents[2] / "docs" / "assets"

PALETTES = {
    "light": {"bg": "#ffffff", "fg": "#1f2328", "muted": "#59636e", "grid": "#d0d7de", "a": "#0969da",
              "b": "#bc4c00", "c": "#1a7f37", "fill": "#0969da", "no": "#d1242f"},
    "dark": {"bg": "#0d1117", "fg": "#c9d1d9", "muted": "#8b949e", "grid": "#30363d", "a": "#58a6ff",
             "b": "#f0883e", "c": "#3fb950", "fill": "#58a6ff", "no": "#f85149"},
}


def svg(width, height, title, desc, body, p):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="t d">\n  <title id="t">{title}</title>\n  <desc id="d">{desc}</desc>\n'
        f"  <style>\n"
        f"    .t {{ font: 600 15px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {p['fg']}; }}\n"
        f"    .b {{ font: 13px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {p['fg']}; }}\n"
        f"    .m {{ font: 12px 'Segoe UI', Helvetica, Arial, sans-serif; fill: {p['muted']}; }}\n"
        f"  </style>\n  <rect width=\"{width}\" height=\"{height}\" fill=\"{p['bg']}\"/>\n{body}</svg>\n"
    )


def families(p):
    x0, y0, w, h = 70, 50, 560, 280
    r = np.linspace(0, 1.6, 321)
    parts = []
    for k in range(9):
        gx = x0 + w * k * 0.2 / 1.6
        parts.append(f'  <line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + h}" stroke="{p["grid"]}"/>')
        parts.append(f'  <text class="m" x="{gx:.1f}" y="{y0 + h + 18}" text-anchor="middle">{k * 0.2:.1f}</text>')
    for k in range(6):
        gy = y0 + h - h * k * 0.2
        parts.append(f'  <line x1="{x0}" y1="{gy:.1f}" x2="{x0 + w}" y2="{gy:.1f}" stroke="{p["grid"]}"/>')
        parts.append(f'  <text class="m" x="{x0 - 8}" y="{gy + 4:.1f}" text-anchor="end">{k * 0.2:.1f}</text>')
    e3 = y0 + h - h * np.exp(-3)
    parts.append(f'  <line x1="{x0}" y1="{e3:.1f}" x2="{x0 + w}" y2="{e3:.1f}" stroke="{p["muted"]}" stroke-dasharray="4 4"/>')
    parts.append(f'  <text class="m" x="{x0 + w - 4}" y="{e3 - 6:.1f}" text-anchor="end">exp(-3) = 0.0498</text>')
    rx = x0 + w / 1.6
    parts.append(f'  <line x1="{rx:.1f}" y1="{y0}" x2="{rx:.1f}" y2="{y0 + h}" stroke="{p["muted"]}" stroke-dasharray="4 4"/>')
    parts.append(f'  <text class="m" x="{rx + 6:.1f}" y="{y0 + 14}">practical range</text>')
    for family, colour, label_r in (("exponential", p["a"], 0.36), ("spherical", p["b"], 0.72), ("gaussian", p["c"], 0.52)):
        rho = correlation(family, r)
        pts = " ".join(f"{x0 + w * ri / 1.6:.1f},{y0 + h - h * v:.1f}" for ri, v in zip(r, rho, strict=True))
        parts.append(f'  <polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        ly = y0 + h - h * float(correlation(family, label_r))
        dy = 22 if family == "exponential" else -10
        parts.append(f'  <text class="b" x="{x0 + w * label_r / 1.6 + 10:.1f}" y="{ly + dy:.1f}" fill="{colour}" '
                     f'style="fill:{colour}">{family}</text>')
    parts.append(f'  <text class="t" x="{x0}" y="30">Correlation against transformed separation r (practical range at r = 1)</text>')
    parts.append(f'  <text class="m" x="{x0 + w / 2}" y="{y0 + h + 38}" text-anchor="middle">r = |diag(1/a) R^T h|</text>')
    return svg(660, 400, "GeoCond correlation families",
               "The exponential, spherical and Gaussian correlations against transformed separation: the exponential "
               "and Gaussian reach exp(-3) at the practical range r = 1, the spherical reaches zero there.",
               "\n".join(parts) + "\n", p)


def tolerance(p):
    ox, oy, scale = 90, 170, 5.0
    t = radians(22.0)
    band = 9.0
    length = 100.0
    parts = ['  <text class="t" x="24" y="30">A directional lag bin: angle tolerance, bandwidth and lag edges</text>']
    # the admitted region: within t of the direction and within the bandwidth of its line
    corner = band / sin(t)
    region = [(0, 0), (corner * cos(t), corner * sin(t)), (length, band), (length, -band), (corner * cos(t), -corner * sin(t))]
    pts = " ".join(f"{ox + x * scale:.1f},{oy - y * scale:.1f}" for x, y in region)
    parts.append(f'  <polygon points="{pts}" fill="{p["fill"]}" fill-opacity="0.12" stroke="{p["fill"]}" stroke-width="1.5"/>')
    for edge in (30.0, 55.0, 80.0):
        half = band if edge >= corner * cos(t) else edge * np.tan(t)
        parts.append(f'  <line x1="{ox + edge * scale:.1f}" y1="{oy - half * scale:.1f}" x2="{ox + edge * scale:.1f}" '
                     f'y2="{oy + half * scale:.1f}" stroke="{p["muted"]}" stroke-dasharray="3 3"/>')
        parts.append(f'  <text class="m" x="{ox + edge * scale:.1f}" y="{oy + band * scale + 18:.1f}" text-anchor="middle">{edge:.0f} m</text>')
    parts.append(f'  <line x1="{ox}" y1="{oy}" x2="{ox + length * scale + 20}" y2="{oy}" stroke="{p["fg"]}" stroke-width="1.5"/>')
    parts.append(f'  <text class="b" x="{ox + length * scale + 24}" y="{oy + 4}">direction d</text>')
    parts.append(f'  <text class="b" x="{ox - 66:.1f}" y="{oy + 34:.1f}">t = 22 degrees</text>')
    parts.append(f'  <line x1="{ox + 92 * scale:.1f}" y1="{oy}" x2="{ox + 92 * scale:.1f}" y2="{oy - band * scale:.1f}" '
                 f'stroke="{p["b"]}" stroke-width="2"/>')
    parts.append(f'  <text class="b" x="{ox + 92 * scale + 6:.1f}" y="{oy - band * scale / 2:.1f}" fill="{p["b"]}" style="fill:{p["b"]}">b</text>')
    for (hx, hy), ok in (((42.0, 6.0), True), ((20.0, 12.5), False), ((70.0, 13.0), False)):
        colour = p["c"] if ok else p["no"]
        parts.append(f'  <line x1="{ox}" y1="{oy}" x2="{ox + hx * scale:.1f}" y2="{oy - hy * scale:.1f}" stroke="{colour}" stroke-width="2"/>')
        parts.append(f'  <circle cx="{ox + hx * scale:.1f}" cy="{oy - hy * scale:.1f}" r="4" fill="{colour}"/>')
    parts.append(f'  <text class="m" x="24" y="{oy + band * scale + 50:.1f}">green: admitted, |h . d| &gt;= |h| cos t and its distance from the line of d is at most b.</text>')
    parts.append(f'  <text class="m" x="24" y="{oy + band * scale + 68:.1f}">red: outside the angle or the band. Lag bin k holds edges[k] &lt;= |h| &lt; edges[k+1]; both tolerances are inclusive.</text>')
    return svg(760, 320, "A directional lag bin",
               "The region of separations a directional variogram admits: within the angle tolerance of the direction "
               "and within the bandwidth of its line, cut into lag bins; one separation inside, two outside.",
               "\n".join(parts) + "\n", p)


def supports(p):
    """A point observation, a line interval and a block target, each with its quadrature, and the nugget rule."""
    from numpy.polynomial.legendre import leggauss

    parts = ['  <text class="t" x="24" y="30">Support-integrated covariance: every support is a known sampling measure</text>']
    # point observation
    parts.append(f'  <circle cx="110" cy="170" r="7" fill="{p["a"]}"/>')
    parts.append('  <text class="b" x="70" y="215">point observation</text>')
    parts.append('  <text class="m" x="70" y="233">one node, weight 1, discrete</text>')
    # line interval with Gauss nodes
    nodes, weights = leggauss(6)
    x0, y0, x1, y1 = 250, 80, 330, 260
    parts.append(f'  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{p["b"]}" stroke-width="3"/>')
    for n, w in zip(nodes, weights, strict=True):
        u = (n + 1) / 2
        parts.append(f'  <circle cx="{x0 + u * (x1 - x0):.1f}" cy="{y0 + u * (y1 - y0):.1f}" r="{3 + 9 * w / 2:.1f}" fill="{p["b"]}"/>')
    parts.append('  <text class="b" x="215" y="285">composite interval</text>')
    parts.append('  <text class="m" x="215" y="303">Gauss nodes, continuous</text>')
    # block target with a 3 by 3 plan of nodes
    bx, by, size = 470, 90, 160
    parts.append(f'  <rect x="{bx}" y="{by}" width="{size}" height="{size}" fill="{p["fill"]}" fill-opacity="0.10" stroke="{p["c"]}" stroke-width="2"/>')
    nodes, weights = leggauss(3)
    for nx, wx in zip(nodes, weights, strict=True):
        for ny, wy in zip(nodes, weights, strict=True):
            parts.append(f'  <circle cx="{bx + (nx + 1) / 2 * size:.1f}" cy="{by + (ny + 1) / 2 * size:.1f}" r="{3 + 5 * wx * wy:.1f}" fill="{p["c"]}"/>')
    parts.append('  <text class="b" x="470" y="285">block target</text>')
    parts.append('  <text class="m" x="470" y="303">product Gauss rule, continuous</text>')
    # averaging arrows
    for (xa, ya), (xb, yb) in (((118, 168), (262, 120)), ((336, 200), (468, 170)), ((118, 172), (468, 200))):
        parts.append(f'  <line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}" stroke="{p["muted"]}" stroke-dasharray="5 4"/>')
    parts.append('  <text class="m" x="24" y="340">C(S, T) = sum over nodes p of S and q of T of w_p w_q C(y_q - x_p). The process nugget enters only between two</text>')
    parts.append('  <text class="m" x="24" y="358">discrete supports at coincident nodes: a point keeps it, a continuous interval or block integrates it to zero.</text>')
    return svg(700, 380, "Support-integrated covariance",
               "A point observation, a composite interval with its Gauss nodes and a block target with a product Gauss "
               "rule; covariance between supports averages the model over their nodes, and the process nugget survives "
               "only between discrete supports at coincident nodes.",
               "\n".join(parts) + "\n", p)


def _axes(parts, p, x0, y0, w, h, xticks, yticks, xfmt, yfmt, xlabel, ylabel):
    for v, pos in xticks:
        parts.append(f'  <line x1="{pos:.1f}" y1="{y0}" x2="{pos:.1f}" y2="{y0 + h}" stroke="{p["grid"]}"/>')
        parts.append(f'  <text class="m" x="{pos:.1f}" y="{y0 + h + 16}" text-anchor="middle">{xfmt(v)}</text>')
    for v, pos in yticks:
        parts.append(f'  <line x1="{x0}" y1="{pos:.1f}" x2="{x0 + w}" y2="{pos:.1f}" stroke="{p["grid"]}"/>')
        parts.append(f'  <text class="m" x="{x0 - 8}" y="{pos + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    parts.append(f'  <text class="m" x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle">{xlabel}</text>')
    parts.append(f'  <text class="m" x="{x0 - 8}" y="{y0 - 8}" text-anchor="end">{ylabel}</text>')


def mik(p):
    """A real multiple-indicator result: raw estimates with an order violation and their bounded isotonic projection."""
    from geocond.covariance import CovarianceComponent, CovarianceModel
    from geocond.kriging import Observations
    from geocond.probability import indicator_kriging
    from geocond.support import point_support

    rng = np.random.default_rng(4)
    x = rng.uniform(0, 50, size=(60, 3))
    z = np.sin(x[:, 0] / 8) + 0.3 * rng.normal(size=60)
    obs = Observations([point_support(q) for q in x], z)
    thresholds = np.quantile(z, [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95])
    models = [CovarianceModel((CovarianceComponent("spherical", r, [[0.25]]),)) for r in (6, 30, 8, 28, 7, 26, 9)]
    grid = [point_support(q) for q in rng.uniform(0, 50, size=(60, 3))]
    out = indicator_kriging(obs, grid, thresholds, models)
    target = int(np.argmax(out.correction_max))
    raw, cdf = out.raw[target], out.cdf[target]
    x0, y0, w, h = 80, 50, 540, 260
    lo, hi = thresholds[0] - 0.15, thresholds[-1] + 0.15
    sx = lambda v: x0 + w * (v - lo) / (hi - lo)
    sy = lambda v: y0 + h - h * (v + 0.1) / 1.2
    parts = [f'  <text class="t" x="{x0}" y="30">Multiple-indicator kriging: raw estimates and the bounded isotonic CDF</text>']
    _axes(parts, p, x0, y0, w, h, [(t, sx(t)) for t in thresholds], [(v, sy(v)) for v in (0.0, 0.25, 0.5, 0.75, 1.0)],
          lambda v: f"{v:.2f}", lambda v: f"{v:.2f}", "threshold (native units)", "P(Z &lt;= t)")
    parts.append(f'  <rect x="{x0}" y="{y0}" width="{sx(thresholds[0]) - x0:.1f}" height="{h}" fill="{p["muted"]}" fill-opacity="0.12"/>')
    parts.append(f'  <rect x="{sx(thresholds[-1]):.1f}" y="{y0}" width="{x0 + w - sx(thresholds[-1]):.1f}" height="{h}" fill="{p["muted"]}" fill-opacity="0.12"/>')
    parts.append(f'  <text class="m" x="{x0 + 6}" y="{y0 + h - 8}">unmodeled tail</text>')
    parts.append(f'  <text class="m" x="{x0 + w - 6}" y="{y0 + h - 8}" text-anchor="end">unmodeled tail</text>')
    pts = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(thresholds, cdf, strict=True))
    parts.append(f'  <polyline points="{pts}" fill="none" stroke="{p["a"]}" stroke-width="2.5"/>')
    for t, a, b in zip(thresholds, raw, cdf, strict=True):
        parts.append(f'  <circle cx="{sx(t):.1f}" cy="{sy(a):.1f}" r="5" fill="none" stroke="{p["b"]}" stroke-width="2"/>')
        parts.append(f'  <circle cx="{sx(t):.1f}" cy="{sy(b):.1f}" r="4" fill="{p["a"]}"/>')
    parts.append(f'  <text class="b" x="{x0}" y="{y0 + h + 58}" fill="{p["b"]}" style="fill:{p["b"]}">open circles: raw ordinary indicator kriging, one covariance per threshold</text>')
    parts.append(f'  <text class="b" x="{x0}" y="{y0 + h + 76}" fill="{p["a"]}" style="fill:{p["a"]}">filled line: corrected CDF (largest change {out.correction_max[target]:.3f}), linear only between thresholds</text>')
    return svg(660, 400, "Multiple-indicator kriging correction",
               "Raw indicator kriging estimates at seven thresholds for one target, which violate the order, and the "
               "bounded isotonic CDF published instead; beyond the first and last thresholds the tails are unmodeled.",
               "\n".join(parts) + "\n", p)


def sgs(p):
    """Three real realizations and the E-type mean along a transect, with the hard data they honour."""
    from geocond.covariance import CovarianceComponent, CovarianceModel
    from geocond.kriging import Observations
    from geocond.simulation import NormalScoreTransform, sequential_gaussian
    from geocond.support import point_support

    xs = np.array([4.0, 17.0, 29.0, 43.0, 58.0, 71.0, 86.0])
    zs = np.array([1.2, 2.9, 2.1, 0.8, 1.9, 3.4, 2.2])
    obs = Observations([point_support([x, 0.0, 0.0]) for x in xs], zs)
    transform = NormalScoreTransform.fit(zs)
    model = CovarianceModel((CovarianceComponent("spherical", 25.0, [[1.0]]),))
    nodes = np.linspace(0, 90, 91)
    result = sequential_gaussian(obs, [point_support([x, 0.0, 0.0]) for x in nodes], model, transform,
                                 realizations=64, seed=11)
    x0, y0, w, h = 70, 50, 560, 250
    sx = lambda v: x0 + w * v / 90
    sy = lambda v: y0 + h - h * (v - 0.5) / 3.2
    parts = [f'  <text class="t" x="{x0}" y="30">Sequential Gaussian simulation along a transect: three of 64 realizations</text>']
    _axes(parts, p, x0, y0, w, h, [(v, sx(v)) for v in range(0, 91, 15)], [(v, sy(v)) for v in (1.0, 2.0, 3.0)],
          lambda v: f"{v:g} m", lambda v: f"{v:g}", "distance along the transect", "value")
    for r, colour in zip(range(3), (p["b"], p["c"], p["muted"]), strict=True):
        pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(nodes, result.native[r], strict=True))
        parts.append(f'  <polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.5" stroke-opacity="0.9"/>')
    pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(nodes, result.etype(), strict=True))
    parts.append(f'  <polyline points="{pts}" fill="none" stroke="{p["a"]}" stroke-width="3"/>')
    for x, z in zip(xs, zs, strict=True):
        parts.append(f'  <circle cx="{sx(x):.1f}" cy="{sy(z):.1f}" r="5" fill="{p["fg"]}"/>')
    parts.append(f'  <text class="b" x="{x0}" y="{y0 + h + 58}">dots: hard data, honoured exactly; thin lines: realizations; thick line: E-type mean of the 64 native realizations</text>')
    return svg(700, 390, "Sequential Gaussian simulation transect",
               "Three sequential Gaussian realizations along a 90 m transect conditioned on seven hard data, which every "
               "realization passes through, and the E-type mean of 64 realizations computed in native units.",
               "\n".join(parts) + "\n", p)


def _raster(parts, grid, x0, y0, cell, colour):
    """Category-1 cells of a 2D grid as one rectangle per horizontal run (x right, y up)."""
    nx, ny = grid.shape
    for j in range(ny):
        i = 0
        while i < nx:
            if grid[i, j] == 1:
                start = i
                while i < nx and grid[i, j] == 1:
                    i += 1
                parts.append(f'  <rect x="{x0 + start * cell:.2f}" y="{y0 + (ny - 1 - j) * cell:.2f}" '
                             f'width="{(i - start) * cell:.2f}" height="{cell + 0.35:.2f}" fill="{colour}" '
                             f'shape-rendering="crispEdges"/>')
            else:
                i += 1


def ds(p):
    """A real training image and three Direct Sampling realizations conditioned on the same hard data."""
    from geocond.direct_sampling import direct_sampling

    n = 64
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ti = (np.sin(2 * np.pi * x / 12 + 1.2 * np.sin(2 * np.pi * y / 20)) > 0.3).astype(float)
    rng = np.random.default_rng(5)
    cells = rng.choice(32 * 32, size=14, replace=False)
    hard_xyz = np.c_[cells % 32, cells // 32, np.zeros(14, int)]
    crop = ti[16:48, 16:48]
    hard = (hard_xyz, crop[hard_xyz[:, 0], hard_xyz[:, 1]])
    reals = [direct_sampling(ti[..., None], (32, 32, 1), variable_kinds=["categorical"], hard_data=hard,
                             max_neighbors=20, threshold=0.05, scan_fraction=0.5, seed=s).realization[:, :, 0, 0]
             for s in (1, 2, 3)]
    parts = ['  <text class="t" x="24" y="28">Direct Sampling: a training image and three conditional realizations</text>']
    size = 160
    panels = [("training image (64 x 64)", ti, 24)] + [(f"realization, seed {s}", r, 24 + (k + 1) * 186)
                                                     for k, (s, r) in enumerate(zip((1, 2, 3), reals, strict=True))]
    for title, grid, px in panels:
        cell = size / grid.shape[0]
        parts.append(f'  <rect x="{px}" y="50" width="{size}" height="{size}" fill="{p["bg"]}" stroke="{p["grid"]}"/>')
        _raster(parts, grid, px, 50, cell, p["a"])
        parts.append(f'  <text class="m" x="{px}" y="{50 + size + 18}">{title}</text>')
        if grid.shape[0] == 32:
            for (i, j, _), v in zip(hard_xyz, hard[1], strict=True):
                cx, cy = px + (i + 0.5) * cell, 50 + (31 - j + 0.5) * cell
                parts.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{p["b"] if v == 1 else p["bg"]}" '
                             f'stroke="{p["b"]}" stroke-width="1.5"/>')
    parts.append('  <text class="m" x="24" y="262">Circles are the 14 hard data (filled: channel, open: background), honoured in every realization; the channel</text>')
    parts.append('  <text class="m" x="24" y="280">geometry between them is copied from the training image. Threshold 0.05, 20 neighbours, half the image scanned.</text>')
    return svg(780, 300, "Direct Sampling realizations",
               "A 64 by 64 categorical training image of meandering channels and three 32 by 32 Direct Sampling "
               "realizations conditioned on the same fourteen hard data, which each realization honours.",
               "\n".join(parts) + "\n", p)


def curvature(p):
    """A three-station survey in a north-vertical plane: the minimum-curvature arc, its stations and the straight chord."""
    from geocond.geometry import Survey

    md = np.array([0.0, 120.0, 260.0])
    survey = Survey([0.0, 0.0, 0.0], md, [0.0, 0.0, 0.0], [-25.0, -55.0, -85.0])
    dense = survey.at(np.linspace(0, md[-1], 200)).points
    mid = survey.at([60.0, 190.0]).points
    x0, y0, scale = 90, 70, 1.25
    X = lambda north: x0 + scale * north
    Y = lambda z: y0 - scale * z
    parts = ['  <text class="t" x="24" y="30">Minimum curvature: positions between stations lie on the arc, not on the chord</text>']
    parts.append(f'  <line x1="{X(0)}" y1="{Y(0)}" x2="{X(0)}" y2="{Y(-215):.1f}" stroke="{p["grid"]}"/>')
    parts.append(f'  <text class="m" x="{X(0) - 8}" y="{Y(-215) + 4:.1f}" text-anchor="end">z</text>')
    parts.append(f'  <line x1="{X(0)}" y1="{Y(0)}" x2="{X(260):.1f}" y2="{Y(0)}" stroke="{p["grid"]}"/>')
    parts.append(f'  <text class="m" x="{X(260):.1f}" y="{Y(0) - 8}" text-anchor="end">north</text>')
    path_d = " ".join(f"{'M' if i == 0 else 'L'}{X(n):.1f},{Y(z):.1f}" for i, (_, n, z) in enumerate(dense))
    parts.append(f'  <path d="{path_d}" fill="none" stroke="{p["a"]}" stroke-width="3"/>')
    stations = survey.station_points
    for (_, n0, z0), (_, n1, z1) in pairwise(stations):
        parts.append(f'  <line x1="{X(n0):.1f}" y1="{Y(z0):.1f}" x2="{X(n1):.1f}" y2="{Y(z1):.1f}" stroke="{p["b"]}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    for (_, n, z), label in zip(stations, ("collar, MD 0", "station, MD 120", "station, MD 260"), strict=True):
        parts.append(f'  <circle cx="{X(n):.1f}" cy="{Y(z):.1f}" r="6" fill="{p["a"]}"/>')
        parts.append(f'  <text class="b" x="{X(n) + 12:.1f}" y="{Y(z) + 4:.1f}">{label}</text>')
    for (_, n, z), (_, na, za), (_, nb, zb) in zip(mid, stations[:-1], stations[1:], strict=True):
        parts.append(f'  <circle cx="{X(n):.1f}" cy="{Y(z):.1f}" r="4.5" fill="{p["c"]}"/>')
        parts.append(f'  <circle cx="{X((na + nb) / 2):.1f}" cy="{Y((za + zb) / 2):.1f}" r="4.5" fill="none" stroke="{p["b"]}" stroke-width="2"/>')
    parts.append(f'  <circle cx="440" cy="300" r="4.5" fill="{p["c"]}"/><text class="b" x="452" y="304">MD half-way, on the arc</text>')
    parts.append(f'  <circle cx="440" cy="324" r="4.5" fill="none" stroke="{p["b"]}" stroke-width="2"/><text class="b" x="452" y="328">chord midpoint (not used)</text>')
    gap = float(np.linalg.norm(mid[1] - (stations[1] + stations[2]) / 2))
    parts.append('  <text class="m" x="24" y="372">Stations at azimuth 0 with dips -25, -55 and -85 degrees. Between the last two stations the half-way point on the arc</text>')
    parts.append(f'  <text class="m" x="24" y="390">is {gap:.2f} m from the chord midpoint; straight-chord interpolation would move every interval centre by up to that much.</text>')
    return svg(760, 410, "Minimum-curvature desurvey",
               "A three-station survey drawn in its north-vertical plane: the circular arcs of minimum curvature through "
               "the stations, the dashed chords between them, and at half-way measured depth the arc position used by "
               "GeoCond next to the chord midpoint it does not use.",
               "\n".join(parts) + "\n", p)


def compositing(p):
    """The acceptance fixture and a gap: source intervals, 2 m composites, their means and coverage."""
    from geocond.compositing import composite_intervals, fixed_boundaries

    rows = [(0.0, 1.0, 2.0), (1.0, 3.0, 5.0), (4.0, 6.0, 3.0)]
    a, b, z = (np.array(c) for c in zip(*rows, strict=True))
    edges = fixed_boundaries(0.0, 6.0, 2.0)
    comps = composite_intervals(a, b, z, edges, source_ids=["s1", "s2", "s3"], min_coverage=1.0)
    x0, unit = 150, 95
    X = lambda depth: x0 + unit * depth
    parts = ['  <text class="t" x="24" y="30">Length-weighted compositing tracks the valid length; a gap is never bridged</text>']
    parts.append('  <text class="b" x="24" y="86">source intervals</text>')
    for lo, hi, value in rows:
        parts.append(f'  <rect x="{X(lo):.1f}" y="66" width="{unit * (hi - lo):.1f}" height="30" fill="{p["fill"]}" fill-opacity="0.18" stroke="{p["a"]}" stroke-width="2"/>')
        parts.append(f'  <text class="b" x="{X((lo + hi) / 2):.1f}" y="86" text-anchor="middle">{value:g}</text>')
    parts.append(f'  <text class="m" x="{X(3.5):.1f}" y="86" text-anchor="middle">gap</text>')
    parts.append('  <text class="b" x="24" y="156">2 m composites</text>')
    for c in comps:
        colour = p["c"] if c.status == "estimated" else p["no"]
        parts.append(f'  <rect x="{X(c.start):.1f}" y="130" width="{unit * (c.end - c.start):.1f}" height="40" fill="none" stroke="{colour}" stroke-width="2"/>')
        mean = f"{c.mean:g}" if c.status == "estimated" else "not estimated"
        parts.append(f'  <text class="b" x="{X((c.start + c.end) / 2):.1f}" y="150" text-anchor="middle">{mean}</text>')
        parts.append(f'  <text class="m" x="{X((c.start + c.end) / 2):.1f}" y="165" text-anchor="middle">coverage {c.coverage:g}</text>')
    for d in range(7):
        parts.append(f'  <line x1="{X(d)}" y1="102" x2="{X(d)}" y2="108" stroke="{p["muted"]}"/>')
        parts.append(f'  <text class="m" x="{X(d)}" y="123" text-anchor="middle">{d} m</text>')
    first = comps[0]
    parts.append(f'  <text class="m" x="24" y="210">[0, 2]: (1 x 2 + 1 x 5) / 2 = {first.mean:g}, numerator {first.numerator:g}, valid length {first.valid_length:g}. [2, 4]: 1 m of 5 over 2 m, coverage 0.5:</text>')
    parts.append('  <text class="m" x="24" y="228">below the declared minimum coverage of 1, so it is kept with its numerator and valid length but has no mean.</text>')
    parts.append('  <text class="m" x="24" y="246">The grade-length integral of the covered length is conserved: 2 + 10 + 6 = 18 = 7 + 5 + 6.</text>')
    return svg(760, 266, "Length-weighted compositing",
               "Three source intervals with values 2, 5 and 3 and a one-metre gap, composited to two-metre intervals: "
               "3.5 with full coverage, an interval with half coverage that is not estimated, and 3 with full coverage.",
               "\n".join(parts) + "\n", p)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode, palette in PALETTES.items():
        (ASSETS / f"correlation-families-{mode}.svg").write_text(families(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"directional-bin-{mode}.svg").write_text(tolerance(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"support-integration-{mode}.svg").write_text(supports(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"mik-correction-{mode}.svg").write_text(mik(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"sgs-transect-{mode}.svg").write_text(sgs(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"direct-sampling-{mode}.svg").write_text(ds(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"minimum-curvature-{mode}.svg").write_text(curvature(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"compositing-{mode}.svg").write_text(compositing(palette), encoding="utf-8", newline="\n")
    print("written to", ASSETS)


if __name__ == "__main__":
    main()
