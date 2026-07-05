"""
Prediction #5: Extra vacuum collective mode — Yukawa fifth-force deviation.

A supersolid vacuum might produce a Yukawa correction to gravity:
  V(r) = -G m1 m2 / r * (1 + alpha * exp(-r / lambda)),  lambda = 1/m_phi

Eot-Wash (Lee et al., PRL 124, 101101, 2020) publishes 95% CL upper bounds on |alpha|
vs Yukawa range lambda. Points above the bound curve are excluded.

Run modes:
  python yukawa_fifth_force_sim.py           # side-by-side: supersolid vs null (default)
  python yukawa_fifth_force_sim.py --mode supersolid
  python yukawa_fifth_force_sim.py --mode null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Approximate 95% CL upper limits on |alpha| vs lambda (m), digitized from the
# combined exclusion contour in Lee et al., Phys. Rev. Lett. 124, 101101 (2020)
# and earlier Eot-Wash compilations. For publication-quality constraints use the
# official plot/data from https://www.npl.washington.edu/eotwash/inverse-square-law
EOTWASH_LAMBDA_M = np.array(
    [
        2.0e-5,
        3.9e-5,
        5.0e-5,
        1.0e-4,
        2.0e-4,
        5.0e-4,
        1.0e-3,
        2.0e-3,
        5.0e-3,
        1.0e-2,
        2.0e-2,
        5.0e-2,
        1.0e-1,
        1.0e0,
    ]
)
EOTWASH_ALPHA_MAX = np.array(
    [
        1.0,
        1.0,
        0.7,
        0.15,
        0.04,
        0.006,
        0.0015,
        5.0e-4,
        1.5e-4,
        6.0e-5,
        2.0e-5,
        5.0e-7,
        2.0e-8,
        1.0e-11,
    ]
)

# Example vacuum-phonon scenarios a supersolid model might predict (illustrative only)
SUPERSOLID_SCENARIOS = [
    ("Long-range mode\nλ=1 m, α=0.1", 1.0, 0.1),
    ("Medium-range mode\nλ=10 cm, α=0.5", 0.1, 0.5),
    ("Short-range mode\nλ=1 cm, α=2", 0.01, 2.0),
    ("Gravity-strength\nλ=100 μm, α=1", 1.0e-4, 1.0),
    ("Weak sub-mm mode\nλ=50 μm, α=0.01", 5.0e-5, 0.01),
]


def alpha_upper_bound(lam: float | np.ndarray) -> float | np.ndarray:
    """Interpolated Eot-Wash 95% CL upper limit on |alpha| at Yukawa range lambda (m)."""
    log_lam = np.log10(np.asarray(lam, dtype=float))
    log_bound = np.interp(
        log_lam,
        np.log10(EOTWASH_LAMBDA_M),
        np.log10(EOTWASH_ALPHA_MAX),
        left=np.log10(EOTWASH_ALPHA_MAX[0]),
        right=np.log10(EOTWASH_ALPHA_MAX[-1]),
    )
    return 10**log_bound


def is_excluded(lam: float, alpha: float) -> bool:
    return abs(alpha) > float(alpha_upper_bound(lam))


def force_ratio_at_r(lam: float, alpha: float, r: np.ndarray) -> np.ndarray:
    """|Delta V / V| = |alpha * exp(-r / lambda)| at separation r."""
    return np.abs(alpha * np.exp(-r / lam))


def exclusion_curve_lam() -> tuple[np.ndarray, np.ndarray]:
    lam = np.logspace(-5, 0, 300)
    return lam, alpha_upper_bound(lam)


def verdict(lam: float, alpha: float) -> str:
    bound = alpha_upper_bound(lam)
    if is_excluded(lam, alpha):
        return f"EXCLUDED (|α|={alpha:.3g} > bound {bound:.3g} at λ={lam:.3g} m)"
    margin = bound / max(abs(alpha), 1e-30)
    return f"allowed (|α|={alpha:.3g} < bound {bound:.3g}; {margin:.1f}× below limit)"


def plot_exclusion_plane(ax: plt.Axes, title: str) -> tuple[np.ndarray, np.ndarray]:
    lam_curve = np.logspace(-5, 0, 300)
    alpha_bound = alpha_upper_bound(lam_curve)
    lam_fill = np.logspace(-5, 0, 200)
    alpha_fill = alpha_upper_bound(lam_fill)
    ax.fill_between(lam_fill, alpha_fill, 1e2, color="#fff3a0", alpha=0.8, label="Excluded (Eöt-Wash 2020, approx.)", zorder=1)
    ax.loglog(lam_curve, alpha_bound, "k-", lw=2, label="95% CL upper bound on |α|", zorder=3)
    ax.set_xlabel(r"Yukawa range $\lambda = 1/m_\phi$ (m)")
    ax.set_ylabel(r"Yukawa strength $|\alpha|$ (relative to gravity)")
    ax.set_title(title)
    ax.set_ylim(1e-12, 1e2)
    ax.set_xlim(1e-5, 1e0)
    ax.grid(alpha=0.3, which="both")
    return lam_curve, alpha_bound


def plot_supersolid_panel(ax: plt.Axes) -> list[str]:
    plot_exclusion_plane(ax, "Supersolid vacuum (hypothetical)\nVacuum-phonon modes vs Eöt-Wash bounds")
    lines = []
    for i, (label, lam, alpha) in enumerate(SUPERSOLID_SCENARIOS):
        color = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"][i]
        marker = "x" if is_excluded(lam, alpha) else "o"
        ax.loglog(lam, abs(alpha), marker, ms=10, mew=2, color=color, zorder=4)
        ax.annotate(
            label.replace("\n", " "),
            (lam, abs(alpha)),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color=color,
        )
        lines.append(f"  {label.replace(chr(10), ' ')}: {verdict(lam, alpha)}")
    ax.legend(fontsize=7, loc="lower left")
    return lines


def plot_null_panel(ax: plt.Axes) -> list[str]:
    plot_exclusion_plane(ax, "Standard QFT + GR (null)\nNo extra Yukawa coupling predicted")
    ax.axhline(1e-12, color="green", ls="--", lw=1.5, label="α → 0 (no fifth force)", zorder=4)
    ax.text(
        3e-4,
        3e-11,
        "Standard physics:\nno vacuum-phonon\nYukawa term",
        fontsize=9,
        color="green",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(fontsize=7, loc="lower left")
    return ["  Standard QFT + GR: α = 0 at all λ → consistent with experiment (not excluded)"]


def plot_force_vs_r(ax: plt.Axes, scenarios: list, title: str) -> None:
    r = np.logspace(-5, 0, 200)
    for label, lam, alpha in scenarios:
        short = label.split("\n")[0]
        ax.loglog(r, force_ratio_at_r(lam, alpha, r), label=short)
    ax.set_xlabel("Separation r (m)")
    ax.set_ylabel(r"$|\Delta V/V| = |\alpha e^{-r/\lambda}|$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)


def save_figure(fig: plt.Figure, filename: str) -> None:
    out = OUTPUT / filename
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def run_compare() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ss_lines = plot_supersolid_panel(axes[0])
    null_lines = plot_null_panel(axes[1])
    fig.suptitle(
        "Prediction #5: Fifth-force constraints — supersolid phonons vs standard physics",
        fontsize=12,
        y=1.02,
    )
    save_figure(fig, "yukawa_fifth_force_sim.png")

  # Secondary plot: force strength vs separation for intuition
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    plot_force_vs_r(
        axes2[0],
        SUPERSOLID_SCENARIOS,
        "Supersolid: force ratio vs separation r",
    )
    axes2[1].set_xscale("log")
    axes2[1].set_yscale("log")
    axes2[1].set_xlim(1e-5, 1e0)
    axes2[1].set_ylim(1e-15, 1e1)
    axes2[1].axhline(1e-15, color="green", ls="--", lw=1.5, label="Standard: ΔV/V → 0")
    axes2[1].text(
        1e-3,
        1e-13,
        "No curves:\nstandard physics\npredicts zero\ndeviation at all r",
        fontsize=9,
        color="green",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    axes2[1].set_xlabel("Separation r (m)")
    axes2[1].set_ylabel(r"$|\Delta V/V|$")
    axes2[1].set_title("Standard QFT (null): no deviation at any r")
    axes2[1].grid(alpha=0.3, which="both")
    axes2[1].legend(fontsize=8)
    save_figure(fig2, "yukawa_fifth_force_vs_r.png")

    print("\nSupersolid vacuum (hypothetical modes):")
    for line in ss_lines:
        print(line)
    n_excluded = sum(1 for _, lam, a in SUPERSOLID_SCENARIOS if is_excluded(lam, a))
    print(f"\n  → {n_excluded}/{len(SUPERSOLID_SCENARIOS)} example modes fall in excluded region")
    print("\nStandard QFT + GR (null):")
    for line in null_lines:
        print(line)


def run_single(mode: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if mode == "supersolid":
        lines = plot_supersolid_panel(ax)
        print("\nSupersolid vacuum (hypothetical modes):")
    else:
        lines = plot_null_panel(ax)
        print("\nStandard QFT + GR (null):")
    for line in lines:
        print(line)
    save_figure(fig, "yukawa_fifth_force_sim.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Yukawa fifth-force simulation (prediction #5)")
    parser.add_argument(
        "--mode",
        choices=["compare", "supersolid", "null"],
        default="compare",
        help="compare: side-by-side (default); supersolid or null: single panel",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare()
    else:
        run_single(args.mode)

    print(
        "\nBounds source: Lee et al., Phys. Rev. Lett. 124, 101101 (2020) [approximate tabulation]."
    )
    print("Official plot: https://www.npl.washington.edu/eotwash/inverse-square-law")


if __name__ == "__main__":
    main()
