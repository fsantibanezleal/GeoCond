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


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode, palette in PALETTES.items():
        (ASSETS / f"correlation-families-{mode}.svg").write_text(families(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"directional-bin-{mode}.svg").write_text(tolerance(palette), encoding="utf-8", newline="\n")
        (ASSETS / f"support-integration-{mode}.svg").write_text(supports(palette), encoding="utf-8", newline="\n")
    print("written to", ASSETS)


if __name__ == "__main__":
    main()
