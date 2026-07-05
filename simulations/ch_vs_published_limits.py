#!/usr/bin/env python3
"""
Plot CH parameter scenarios against published experimental limits (real units).

  python ch_vs_published_limits.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

C = 299_792_458.0
HBAR = 1.054_571_817e-34
G_MEAS = 6.674_30e-11

# Eöt-Wash 2020 approximate 95% CL upper |α| vs λ [m]
EOTWASH_LAM = np.array([2e-5, 3.9e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1.0])
EOTWASH_ALPHA = np.array([1.0, 1.0, 0.15, 0.006, 0.0015, 6e-5, 2e-8, 1e-11])

# Fermi GRB090510-inspired quadratic dispersion: β_max vs energy at D=2 Gpc
DISTANCE_GPC = 2.0
DISTANCE_M = DISTANCE_GPC * 3.086e25
E_QG_GEV = 1.0e11
GEV_TO_J = 1.602176634e-10
BETA_FERMI_BOUND = 1.5 * C**2 / (E_QG_GEV * GEV_TO_J) ** 2


def ch_rho_in(xi: float) -> float:
    return C**2 * xi / G_MEAS


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel A: Eöt-Wash ---
    ax = axes[0]
    lam = np.logspace(-5, 0, 300)
    alpha_bound = np.interp(
        np.log10(lam), np.log10(EOTWASH_LAM), np.log10(EOTWASH_ALPHA),
        left=np.log10(EOTWASH_ALPHA[0]), right=np.log10(EOTWASH_ALPHA[-1]),
    )
    alpha_bound = 10**alpha_bound
    ax.fill_between(lam, alpha_bound, 1e2, color="#fff3a0", alpha=0.8, label="Excluded (Eöt-Wash ~2020)")
    ax.loglog(lam, alpha_bound, "k-", lw=2)

    scenarios = [
        ("CH phonon: λ=1 m, α=0.1", 1.0, 0.1, "x"),
        ("CH phonon: λ=1 cm, α=2", 0.01, 2.0, "x"),
        ("CH weak: λ=50 μm, α=0.01", 5e-5, 0.01, "o"),
    ]
    for label, lam0, alpha, mk in scenarios:
        excluded = alpha > np.interp(lam0, EOTWASH_LAM, EOTWASH_ALPHA)
        color = "#d62728" if excluded else "#2ca02c"
        ax.loglog(lam0, alpha, mk, ms=10, mew=2, color=color)
        ax.annotate(label.split(": ")[1], (lam0, alpha), fontsize=7, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(r"Yukawa range $\lambda$ (m)")
    ax.set_ylabel(r"$|\alpha|$ relative to gravity")
    ax.set_title("Real bound: fifth-force (Eöt-Wash)\nCH scenarios in actual meters")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # --- Panel B: ρ_in vs ξ with real G ---
    ax = axes[1]
    xi = np.logspace(-6, 0, 200)
    rho_in = C**2 * xi / G_MEAS
    eps = 9.9e-27 / rho_in

    ax.loglog(xi * 1e3, rho_in, "b-", lw=2, label=r"$\rho_{\rm in} = c^2 \xi / G$")
    ax2 = ax.twinx()
    ax2.loglog(xi * 1e3, eps, "r--", lw=2, label=r"$\varepsilon = \rho_\Lambda / \rho_{\rm in}$")
    ax.set_xlabel(r"Healing length $\xi$ (mm)")
    ax.set_ylabel(r"$\rho_{\rm in}$ (kg/m³)", color="b")
    ax2.set_ylabel(r"Gravitating fraction $\varepsilon$", color="r")
    ax.set_title(f"CH real units: $\\rho_{{\\rm in}}$ & $\\varepsilon$ vs $\\xi$\n($G = {G_MEAS:.3e}$, $\\rho_\\Lambda \\sim 10^{{-26}}$ kg/m³)")

    for x_mm, lab in [(1e-6, "1 nm"), (0.15, "150 nm"), (1, "1 mm")]:
        xi_m = x_mm * 1e-3
        ax.axvline(x_mm, color="gray", ls=":", alpha=0.6)
        ax.text(x_mm, 1e10, lab, rotation=90, fontsize=7, va="bottom")

    ax.grid(alpha=0.3, which="both")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper left")

    fig.suptitle("CH theory vs real experimental / cosmological scales (SI units)", fontsize=12, y=1.02)
    out = OUTPUT / "ch_vs_published_limits.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    print(f"Fermi-inspired β bound (quadratic): β ≲ {BETA_FERMI_BOUND:.2e}")
    print("Run: python ch_real_units_analysis.py for full text report")


if __name__ == "__main__":
    main()
