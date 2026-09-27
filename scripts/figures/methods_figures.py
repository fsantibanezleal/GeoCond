#!/usr/bin/env python3
"""Write the method figures of docs/methods as light and dark SVG pairs (standard library only, plus GeoCond).

Usage: python scripts/figures/methods_figures.py   (from the repository root)
"""

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


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode, palette in PALETTES.items():
        (ASSETS / f"correlation-families-{mode}.svg").write_text(families(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"directional-bin-{mode}.svg").write_text(tolerance(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"support-integration-{mode}.svg").write_text(supports(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"mik-correction-{mode}.svg").write_text(mik(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"sgs-transect-{mode}.svg").write_text(sgs(palette), encoding="utf-8", newline="\n")
    print("written to", ASSETS)


if __name__ == "__main__":
    main()
