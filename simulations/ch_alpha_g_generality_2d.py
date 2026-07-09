"""
2D oblate generality — compare N_hydro on equator vs pole rays.

Links ch_gpe_gravity_axisymmetric.py (true 2D inner + exterior rays) to the
1D equatorial sweep in ch_alpha_g_generality_sweep.py.

Theory question: α_G^hydro,ref is grain-universal; does exterior Newton stay
~1 on both rays when the sink is oblate?

  python ch_alpha_g_generality_2d.py
  python ch_alpha_g_generality_2d.py --quick
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_alpha_g_generality_sweep import evaluate_point as evaluate_1d
from ch_defect_microphysics import gamma0_from_microphysics
from ch_dispersion_core import CHParams
from ch_gpe_gravity import calibrate_g0_matched, sigma_effective_oblate
from ch_gpe_gravity_axisymmetric import solve_oblate_inner_budget, solve_oblate_with_exterior

OUTPUT = Path(__file__).parent / "output"


@dataclass
class Generality2DPoint:
    xi_m: float
    rs_hat: float
    sigma_par: float
    sigma_perp: float
    sigma_eff: float
    gamma0: float
    newton_1d: float
    newton_eq_2d: float
    newton_pol_2d: float
    newton_pol_over_eq: float
    pole_rho_over_eq: float
    m_inner_2d_kg: float
    rho_pol_over_eq_R1: float
    inner_residual: float
    alpha_g_hydro_ref: float


def evaluate_2d_point(
    xi_m: float,
    rs_hat: float,
    sigma_par: float,
    sigma_perp: float,
    *,
    nr: int = 72,
    nz: int = 72,
    n_iter: int = 10000,
) -> Generality2DPoint:
    ch = CHParams(xi=xi_m, alpha_g=1.0)
    p1d = evaluate_1d(xi_m, rs_hat, sigma_par, sigma_perp)

    g0 = gamma0_from_microphysics(ch, sigma_perp)
    if not np.isfinite(g0) or g0 <= 0:
        g0, _, _, _ = calibrate_g0_matched(rs_hat, sigma_hat=sigma_perp)

    demo = solve_oblate_with_exterior(
        ch,
        g0,
        sigma_par,
        sigma_perp,
        rs_hat,
        nr=nr,
        nz=nz,
        n_iter=n_iter,
    )

    inner = solve_oblate_inner_budget(
        ch,
        g0,
        sigma_par,
        sigma_perp,
        rs_hat,
        grid_levels=(max(60, nr - 12), nr, min(nr + 36, 144)),
        n_iter_base=n_iter,
    )

    return Generality2DPoint(
        xi_m=xi_m,
        rs_hat=rs_hat,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        sigma_eff=sigma_effective_oblate(sigma_par, sigma_perp),
        gamma0=g0,
        newton_1d=p1d.newton_at_hydro_ref,
        newton_eq_2d=demo["newton_eq"],
        newton_pol_2d=demo["newton_pol"],
        newton_pol_over_eq=demo["newton_pole_over_eq"],
        pole_rho_over_eq=inner.rho_pol_over_eq_R1,
        m_inner_2d_kg=inner.m_inner_kg,
        rho_pol_over_eq_R1=inner.rho_pol_over_eq_R1,
        inner_residual=inner.residual,
        alpha_g_hydro_ref=p1d.alpha_g_hydro_ref,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    xi = 50e-9
    rs_vals = [1.0] if args.quick else [0.5, 1.0, 2.0]
    sig_pairs = (
        [(0.6, 0.6), (0.6, 1.2), (1.2, 0.6)]
        if args.quick
        else [
            (0.6, 0.6),
            (0.8, 0.4),
            (0.4, 0.8),
            (0.6, 1.2),
            (1.2, 0.6),
        ]
    )
    nr, nz, n_iter = (60, 60, 6000) if args.quick else (72, 72, 10000)

    rows = [
        evaluate_2d_point(xi, rs, sp, sperp, nr=nr, nz=nz, n_iter=n_iter)
        for rs in rs_vals
        for sp, sperp in sig_pairs
    ]

    std_1d = np.std([p.newton_1d for p in rows])
    std_eq = np.std([p.newton_eq_2d for p in rows])
    std_pol = np.std([p.newton_pol_2d for p in rows])
    max_asym = max(abs(p.newton_pol_over_eq - 1.0) for p in rows)

    report = "\n".join(
        [
            "CH alpha_G generality — 2D oblate (equator vs pole rays)",
            f"n = {len(rows)}",
            f"alpha_G^hydro,ref (grain): {rows[0].alpha_g_hydro_ref:.6e}",
            f"std(N_1d equatorial slice) = {std_1d:.4f}",
            f"std(N_2d equator ray)       = {std_eq:.4f}",
            f"std(N_2d pole ray)          = {std_pol:.4f}",
            f"max |N_pol/N_eq - 1|        = {max_asym:.4f}",
            "",
            "CH reading:",
            "  Inner depletion can be pole/equator asymmetric (rho ratio varies).",
            "  Exterior hydrostatic Newton at grain alpha_G can still be ~1 on both",
            "  rays when the matched Schwarzschild tail is appended on each ray.",
            "",
            "Per-point:",
        ]
        + [
            f"  rs={p.rs_hat} sig=({p.sigma_par},{p.sigma_perp}): "
            f"N_1d={p.newton_1d:.3f} N_eq={p.newton_eq_2d:.3f} "
            f"N_pol={p.newton_pol_2d:.3f} rho_pol/eq@R=xi={p.rho_pol_over_eq_R1:.3f} "
            f"M_inner={p.m_inner_2d_kg:.3e} res={p.inner_residual:.2e}"
            for p in rows
        ]
    )
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_alpha_g_generality_2d.txt").write_text(report + "\n")
    fields = list(Generality2DPoint.__dataclass_fields__.keys())
    with (OUTPUT / "ch_alpha_g_generality_2d.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in rows:
            w.writerow({k: getattr(p, k) for k in fields})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sig_eff = [p.sigma_eff for p in rows]
    axes[0].scatter(sig_eff, [p.newton_eq_2d for p in rows], label="2D equator", c="C0")
    axes[0].scatter(sig_eff, [p.newton_pol_2d for p in rows], label="2D pole", c="C3", marker="s")
    axes[0].scatter(sig_eff, [p.newton_1d for p in rows], label="1D slice", c="k", marker="x", alpha=0.6)
    axes[0].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel(r"$\sigma_{\mathrm{eff}}/\xi$")
    axes[0].set_ylabel(r"$N_{\mathrm{hydro}}$ @ grain ref")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Exterior Newton: 2D rays vs 1D slice")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        [p.rho_pol_over_eq_R1 for p in rows],
        [p.newton_pol_over_eq for p in rows],
        c=[p.rs_hat for p in rows],
        cmap="viridis",
    )
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[1].axvline(1.0, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel(r"$\rho_{\mathrm{pole,mid}}/\rho_{\mathrm{eq,mid}}$ (inner)")
    axes[1].set_ylabel(r"$N_{\mathrm{pol}}/N_{\mathrm{eq}}$ (exterior)")
    axes[1].set_title("Inner asymmetry vs exterior Newton asymmetry")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT / "ch_alpha_g_generality_2d.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUTPUT}/ch_alpha_g_generality_2d.*")


if __name__ == "__main__":
    main()
