"""
1D GPE Casimir-gap solver — boundary amplification of |∇ρ| (CH Prediction #7).

Solves the stationary Gross–Pitaevskii equation between parallel plates
(ψ=0 at walls), scans gap separation d, and plots |∇ρ|/|∇ρ|_c vs d.

This is the numerical "GPE step" described in docs/ch-gradient-threshold-experiment.md §6.

  python ch_gpe_casimir_gap.py
  python ch_gpe_casimir_gap.py --xi 50e-9 --d-min 10e-9 --d-max 1000e-9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams, chi
from ch_gpe_core import (
    GPE1DResult,
    scan_gap_separations,
    tidal_grad_ratio,
    wall_scaling_ratio,
)

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

M_EARTH = 5.9722e24
R_EARTH = 6.371e6


def predict_alpha_eff(result: GPE1DResult, alpha_max: float = 0.12) -> float:
    """CH ripple amplitude from bulk χ(|∇ρ|) (central-half maximum)."""
    g_mid = result.grad_ratio_mid * result.grad_rho_crit
    return alpha_max * float(chi(g_mid, result.grad_rho_crit))


def plot_results(
    ch: CHParams,
    results: list[GPE1DResult],
    alpha_max: float,
    out_png: Path,
) -> None:
    d_nm = np.array([r.d_m for r in results]) * 1e9
    ratio_mid = np.array([r.grad_ratio_mid for r in results])
    ratio_wall = np.array([r.grad_ratio_wall for r in results])
    chi_mid = np.array([r.chi_mid for r in results])
    chi_wall = np.array([r.chi_wall for r in results])
    alpha_mid = alpha_max * chi_mid
    wall_est = np.array([wall_scaling_ratio(ch, r.d_m) for r in results])
    tidal = tidal_grad_ratio(ch, M_EARTH, R_EARTH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].semilogx(d_nm, ratio_wall, "o-", label=r"GPE wall $|\nabla\rho|/|\nabla\rho|_c$")
    axes[0, 0].semilogx(d_nm, ratio_mid, "s--", label=r"GPE bulk (central half)")
    axes[0, 0].loglog(d_nm, wall_est, ":", color="gray", label=r"Scaling $\xi/d$")
    axes[0, 0].axhline(1.0, color="k", ls="--", lw=0.8, label=r"Threshold ($\chi=1$)")
    axes[0, 0].axhline(tidal, color="orange", ls=":", label=rf"Earth tidal ({tidal:.1e})")
    axes[0, 0].set_xlabel("Gap separation d [nm]")
    axes[0, 0].set_ylabel(r"$|\nabla\rho| / |\nabla\rho|_c$")
    axes[0, 0].set_title("Boundary amplification from 1D GPE")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(alpha=0.3, which="both")

    axes[0, 1].semilogx(d_nm, chi_wall, "o-", label=r"$\chi$ at wall")
    axes[0, 1].semilogx(d_nm, chi_mid, "s--", label=r"$\chi_{\mathrm{bulk}}$")
    axes[0, 1].set_xlabel("Gap d [nm]")
    axes[0, 1].set_ylabel(r"$\chi$")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].semilogx(d_nm, alpha_mid, "o-", color="C3")
    axes[1, 0].set_xlabel("Gap d [nm]")
    axes[1, 0].set_ylabel(r"$\alpha_{\mathrm{eff}}$ (bulk coupling)")
    axes[1, 0].set_title(rf"Predicted Casimir ripple ($\alpha_{{\max}}$={alpha_max})")
    axes[1, 0].grid(alpha=0.3)

    # Example density profile for smallest, middle, largest d
    pick = [0, len(results) // 2, -1]
    ax = axes[1, 1]
    for i in pick:
        r = results[i]
        z_nm = r.z_hat * r.xi_m * 1e9
        ax.plot(z_nm, r.rho_norm, label=rf"d={r.d_m*1e9:.0f} nm")
    ax.set_xlabel("Position z [nm]")
    ax.set_ylabel(r"$\rho / \rho_{\mathrm{in}}$")
    ax.set_title(r"Density profiles ($|ψ|^2$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        rf"CH GPE Casimir gap ($\xi$={ch.xi:.1e} m, $|\nabla\rho|_c$={ch.grad_rho_crit:.2e} kg/m⁴)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(ch: CHParams, results: list[GPE1DResult], alpha_max: float, path: Path) -> None:
    tidal = tidal_grad_ratio(ch, M_EARTH, R_EARTH)
    with path.open("w") as f:
        f.write("CH 1D GPE Casimir gap scan\n")
        f.write(f"xi [m] = {ch.xi:.6e}\n")
        f.write(f"rho_in [kg/m^3] = {ch.rho_in:.6e}\n")
        f.write(f"|grad rho|_c [kg/m^4] = {ch.grad_rho_crit:.6e}\n")
        f.write(f"Earth tidal |grad rho|/|grad rho|_c = {tidal:.6e}\n\n")
        f.write("d[nm]  d/xi   grad_wall  grad_mid   chi_wall   chi_mid   alpha_mid\n")
        for r in results:
            f.write(
                f"{r.d_m*1e9:8.1f} {r.d_hat:8.2e} {r.grad_ratio_wall:10.3e} "
                f"{r.grad_ratio_mid:10.3e} {r.chi_wall:9.3e} {r.chi_mid:9.3e} "
                f"{predict_alpha_eff(r, alpha_max):13.3e}\n"
            )
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="1D GPE Casimir-gap |∇ρ| scan")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=10e-9, help="Min gap [m]")
    parser.add_argument("--d-max", type=float, default=1000e-9, help="Max gap [m]")
    parser.add_argument("--n-d", type=int, default=25, help="Number of gap points")
    parser.add_argument("--alpha-max", type=float, default=0.12, help="Ripple amplitude scale")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_array = np.logspace(np.log10(args.d_min), np.log10(args.d_max), args.n_d)

    print("=" * 60)
    print("CH 1D GPE Casimir gap — boundary |∇ρ| scan")
    print("=" * 60)
    print(f"ξ = {ch.xi:.3e} m")
    print(f"|∇ρ|_c = {ch.grad_rho_crit:.3e} kg/m⁴")
    print(f"Earth tidal ratio = {tidal_grad_ratio(ch, M_EARTH, R_EARTH):.3e}")
    print(f"Scanning d = {args.d_min*1e9:.1f} – {args.d_max*1e9:.1f} nm ({args.n_d} points)")

    results = scan_gap_separations(ch, d_array)

    r_lo = results[0]
    r_hi = results[-1]
    print(f"\nAt d = {r_lo.d_m*1e9:.1f} nm: wall |∇ρ|/|∇ρ|_c = {r_lo.grad_ratio_wall:.3f}, χ_wall = {r_lo.chi_wall:.3e}")
    print(f"At d = {r_hi.d_m*1e9:.1f} nm: wall |∇ρ|/|∇ρ|_c = {r_hi.grad_ratio_wall:.3f}, bulk χ ≈ {r_hi.chi_mid:.3e}")

    if max(r.chi_mid for r in results) < 0.01:
        print("\n→ Midpoint χ ≈ 0 for all gaps at this ξ (wide-gap Casimir measures near walls, not center).")
        print("  Wall χ ~ {:.2f} (Thomas–Fermi); turn-on at gap center needs d ~ O(ξ).".format(
            max(r.chi_wall for r in results)
        ))

    plot_results(ch, results, args.alpha_max, OUTPUT / "ch_gpe_casimir_gap.png")
    write_report(ch, results, args.alpha_max, OUTPUT / "ch_gpe_casimir_gap.txt")


if __name__ == "__main__":
    main()
