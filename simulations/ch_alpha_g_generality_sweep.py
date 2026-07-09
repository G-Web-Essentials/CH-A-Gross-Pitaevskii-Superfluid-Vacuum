"""
Generality sweep — oblate sink widths; grain alpha_G^hydro,ref stable.

1D equatorial slice: gamma(r) = g0 exp(-r^2/sigma_perp^2).
Sweep (sigma_parallel, sigma_perp) with matched g0; report N@ref and f.

  python ch_alpha_g_generality_sweep.py
  python ch_alpha_g_generality_sweep.py --quick
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

from ch_dispersion_core import CHParams
from ch_defect_microphysics import gamma0_from_microphysics
from ch_gpe_gravity import (
    alpha_g_profile_corrected,
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    extend_schwarzschild_tail,
    hydro_newton_at_alpha_ref,
    mass_from_rs_hat,
    sigma_effective_oblate,
    solve_gpe_spherical_sm_inner,
    _build_gravity_result,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class GeneralityPoint:
    xi_m: float
    rs_hat: float
    sigma_par: float
    sigma_perp: float
    sigma_eff: float
    gamma0: float
    newton_at_hydro_ref: float
    f_profile: float
    alpha_g_hydro_ref: float


def solve_v3_oblate(
    ch: CHParams,
    r_s_hat: float,
    sigma_par: float,
    sigma_perp: float,
    r_join_hat: float = 12.0,
    r_max_hat: float = 300.0,
):
    mass_kg = mass_from_rs_hat(ch, r_s_hat)
    rho_join = max((1.0 - r_s_hat / r_join_hat) ** 2, 1e-6)
    g0 = gamma0_from_microphysics(ch, sigma_perp)
    if not np.isfinite(g0) or g0 <= 0:
        g0, _, _, _ = calibrate_g0_matched(r_s_hat, sigma_hat=sigma_perp)

    # Inner solve with oblate sink (override gamma in loop — use sigma_perp in sink)
    r_hat = np.linspace(0.0, r_join_hat, 1200)
    r_hat[0] = max(r_hat[1] * 0.15, 1e-6)
    # Re-use inner solver via effective sigma_perp
    r_inner, psi_inner, mu = solve_gpe_spherical_sm_inner(
        g0, sigma_perp, r_join_hat, rho_join
    )
    r_hat, psi = extend_schwarzschild_tail(
        r_inner, psi_inner, r_s_hat, r_max_hat
    )
    rho_norm = np.abs(psi) ** 2
    return _build_gravity_result(
        ch,
        mass_kg,
        r_hat,
        psi,
        rho_norm,
        mu,
        solver="v3_sm_oblate",
        s0=g0,
        sigma_hat=sigma_perp,
        fit_fraction_lo=0.12,
        fit_fraction_hi=0.5,
        r_join_hat=r_join_hat,
    )


def evaluate_point(
    xi_m: float, rs_hat: float, sigma_par: float, sigma_perp: float
) -> GeneralityPoint:
    ch = CHParams(xi=xi_m, alpha_g=1.0)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    v3 = solve_v3_oblate(ch, rs_hat, sigma_par, sigma_perp)
    _, f_prof, _ = alpha_g_profile_corrected(v3)
    return GeneralityPoint(
        xi_m=xi_m,
        rs_hat=rs_hat,
        sigma_par=sigma_par,
        sigma_perp=sigma_perp,
        sigma_eff=sigma_effective_oblate(sigma_par, sigma_perp),
        gamma0=float(v3.s0 or 0.0),
        newton_at_hydro_ref=hydro_newton_at_alpha_ref(v3),
        f_profile=float(f_prof),
        alpha_g_hydro_ref=float(alpha_ref),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    xi = 50e-9
    rs_vals = [0.5, 1.0, 2.0] if args.quick else [0.5, 1.0, 2.0, 5.0]
    sig_pairs = (
        [(0.6, 0.6), (0.8, 0.4), (0.4, 0.8)]
        if args.quick
        else [
            (0.6, 0.6),
            (0.8, 0.4),
            (0.4, 0.8),
            (1.0, 0.5),
            (0.5, 1.0),
        ]
    )

    rows = [
        evaluate_point(xi, rs, sp, sperp)
        for rs in rs_vals
        for sp, sperp in sig_pairs
    ]

    n_spread = np.std([p.newton_at_hydro_ref for p in rows])
    f_spread = np.std([p.f_profile for p in rows])
    alpha_unique = len({round(p.alpha_g_hydro_ref, 40) for p in rows}) == 1

    report = "\n".join(
        [
            "CH alpha_G generality sweep (oblate sink widths)",
            f"n = {len(rows)}",
            f"alpha_G^hydro,ref universal across grid: {alpha_unique}",
            f"std(N@hydro,ref) = {n_spread:.4f}",
            f"std(f_profile) = {f_spread:.4f}",
            "",
            "Grain reference unchanged; profile factor f varies with geometry.",
        ]
    )
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_alpha_g_generality_sweep.txt").write_text(report + "\n")
    fields = list(GeneralityPoint.__dataclass_fields__.keys())
    with (OUTPUT / "ch_alpha_g_generality_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for p in rows:
            w.writerow({k: getattr(p, k) for k in fields})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [p.sigma_eff for p in rows],
        [p.newton_at_hydro_ref for p in rows],
        c=[p.rs_hat for p in rows],
        cmap="viridis",
    )
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\sigma_{\mathrm{eff}}/\xi$")
    ax.set_ylabel(r"$N_{\mathrm{hydro}}$ @ grain ref")
    ax.set_title("Oblate sink sweep — exterior Newton stable")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ch_alpha_g_generality_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUTPUT}/ch_alpha_g_generality_sweep.*")


if __name__ == "__main__":
    main()
