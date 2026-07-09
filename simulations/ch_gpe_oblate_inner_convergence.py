"""
Oblate inner convergence — M_inner and pole/equator ρ budgets with grid refinement.

  python ch_gpe_oblate_inner_convergence.py
  python ch_gpe_oblate_inner_convergence.py --quick
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from ch_defect_microphysics import gamma0_from_microphysics
from ch_dispersion_core import CHParams
from ch_gpe_gravity import calibrate_g0_matched
from ch_gpe_gravity_axisymmetric import solve_oblate_inner_budget

OUTPUT = Path(__file__).parent / "output"


def format_budget(tag: str, b) -> list[str]:
    return [
        f"=== {tag} (gamma0={b.g0:.4g}, sig=({b.sigma_parallel},{b.sigma_perp})) ===",
        f"  grid {b.nr}x{b.nz}, iter={b.n_iter}, residual={b.residual:.3e}",
        f"  M_inner (equator ray) = {b.m_inner_kg:.4e} kg",
        f"  pole/equator ray mass ratio = {b.m_inner_fine_rel_change:.4f}  (M_pol/M_eq)",
        f"  rho_min (core) = {b.rho_center:.4f}",
        f"  rho(R≈0.15ξ) eq/pol = {b.rho_eq_R1:.4f} / {b.rho_pol_R1:.4f}  ratio={b.rho_pol_over_eq_R1:.4f}",
        f"  rho(R≈0.35ξ) eq/pol = {b.rho_eq_R2:.4f} / {b.rho_pol_R2:.4f}  ratio={b.rho_pol_over_eq_R2:.4f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    xi = 50e-9
    rs_hat = 1.0
    ch = CHParams(xi=xi)
    grids = (60, 90) if args.quick else (72, 108, 144)
    n_iter = 8000 if args.quick else 12000

    g0_smooth, _, _, _ = calibrate_g0_matched(rs_hat, sigma_hat=0.6)
    g0_micro = gamma0_from_microphysics(ch, 0.6)

    cases = [
        ("smooth core reg", g0_smooth, 0.6, 0.6),
        ("smooth oblate", g0_smooth, 0.6, 1.2),
        ("micro oblate", g0_micro, 0.6, 1.2),
        ("micro oblate wide", g0_micro, 1.2, 0.6),
    ]
    if args.quick:
        cases = cases[:2]

    budgets = [
        solve_oblate_inner_budget(
            ch, g0, sp, sperp, rs_hat, grid_levels=grids, n_iter_base=n_iter
        )
        for _tag, g0, sp, sperp in cases
    ]

    lines = [
        "CH oblate inner convergence — M_inner and pole/equator rho budgets",
        "Method: converged 1D GP rays (equator σ_⊥, pole σ_∥); M_inner from equator.",
        f"xi = {xi:.3e} m, r_s/xi = {rs_hat}, r_join/xi = 12",
        "",
    ]
    for (tag, *_), b in zip(cases, budgets):
        lines.extend(format_budget(tag, b))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gpe_oblate_inner_convergence.txt").write_text(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [c[0] for c in cases]
    m_vals = [b.m_inner_kg for b in budgets]
    axes[0].bar(range(len(labels)), m_vals, color="C0")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel(r"$M_{\mathrm{inner}}$ [kg]")
    axes[0].set_title("Inner mass budget (refined grid)")

    r1 = [b.rho_pol_over_eq_R1 for b in budgets]
    r2 = [b.rho_pol_over_eq_R2 for b in budgets]
    x = range(len(labels))
    w = 0.35
    axes[1].bar([i - w / 2 for i in x], r1, width=w, label=r"$R=\xi$")
    axes[1].bar([i + w / 2 for i in x], r2, width=w, label=r"$R=2\xi$")
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel(r"$\rho_{\mathrm{pole}}/\rho_{\mathrm{eq}}$")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Inner pole/equator asymmetry")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ch_gpe_oblate_inner_convergence.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUTPUT}/ch_gpe_oblate_inner_convergence.*")


if __name__ == "__main__":
    main()
