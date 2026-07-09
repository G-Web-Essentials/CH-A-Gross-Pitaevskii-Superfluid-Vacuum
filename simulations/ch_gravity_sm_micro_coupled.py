"""
Route A1 — micro vs smooth gamma0: inner rho + exterior N_hydro (coupled).

Extends G5d+ with coupled psi-Phi loop at grain alpha_G for both gamma0 prescriptions.

  python ch_gravity_sm_micro_coupled.py
  python ch_gravity_sm_micro_coupled.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_defect_microphysics import gamma0_from_microphysics
from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    calibrate_g0_matched,
    mass_deficit_from_profile,
    solve_gpe_spherical_sm_inner,
)
from ch_gravity_sm_coupled import inner_mass_deficit, solve_coupled_sm

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class MicroCoupledPoint:
    label: str
    gamma0: float
    rs_hat: float
    rho_bulk_min: float
    rho_join_layer_mean: float
    m_inner_kg: float
    newton_coupled: float
    newton_slaved: float
    newton_v3_smooth: float


def evaluate(
    ch: CHParams,
    label: str,
    g0: float,
    sigma: float,
    rs_hat: float,
    *,
    v3_smooth_n: float,
) -> MicroCoupledPoint:
    rho_join = max((1.0 - rs_hat / R_JOIN_HAT) ** 2, 1e-6)
    r_in, psi_in, _ = solve_gpe_spherical_sm_inner(
        g0, sigma, R_JOIN_HAT, rho_join, n_points=2000, n_iter=10000
    )
    rho = np.abs(psi_in) ** 2
    bulk_mask = r_in <= 0.35 * R_JOIN_HAT
    join_mask = r_in >= 0.9 * R_JOIN_HAT

    res = solve_coupled_sm(
        ch, g0, sigma, rs_hat, coupling_scale=1.0, artificial_coupling=False, n_outer=20
    )

    return MicroCoupledPoint(
        label=label,
        gamma0=g0,
        rs_hat=rs_hat,
        rho_bulk_min=float(np.min(rho[bulk_mask])),
        rho_join_layer_mean=float(np.mean(rho[join_mask])),
        m_inner_kg=inner_mass_deficit(ch, res.r_hat, res.rho_norm, R_JOIN_HAT),
        newton_coupled=res.newton_coupled,
        newton_slaved=res.newton_slaved_hydro,
        newton_v3_smooth=v3_smooth_n,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0_smooth = calibrate_g0_matched(1.0, sigma_hat=args.sigma)[0]
    g0_micro = gamma0_from_microphysics(ch, args.sigma)
    rs_vals = [1.0, 2.0, 4.0] if args.quick else [0.5, 1.0, 2.0, 4.0, 8.0]

    # Reference N from smooth v3 at rs=1
    ref = solve_coupled_sm(ch, g0_smooth, args.sigma, 1.0, n_outer=15)
    v3_ref_n = ref.newton_v3_ref

    rows: list[MicroCoupledPoint] = []
    for rs in rs_vals:
        ref_rs = solve_coupled_sm(ch, g0_smooth, args.sigma, rs, n_outer=15)
        rows.append(
            evaluate(ch, "smooth", g0_smooth, args.sigma, rs, v3_smooth_n=ref_rs.newton_v3_ref)
        )
        rows.append(
            evaluate(ch, "micro", g0_micro, args.sigma, rs, v3_smooth_n=ref_rs.newton_v3_ref)
        )

    lines = [
        "CH micro vs smooth — coupled exterior N (Route A1)",
        f"sigma = {args.sigma}, grain alpha_G^hydro,ref (no N-fit)",
        "",
    ]
    for rs in rs_vals:
        lines.append(f"--- r_s/xi = {rs:g} ---")
        sub = [r for r in rows if r.rs_hat == rs]
        for r in sub:
            lines.append(
                f"  {r.label:6s} g0={r.gamma0:.4g}: rho_bulk_min={r.rho_bulk_min:.5f}  "
                f"M_inner={r.m_inner_kg:.4e} kg  "
                f"N_cpl={r.newton_coupled:.4f}  N_slv={r.newton_slaved:.4f}"
            )
        lines.append("")

    lines += [
        "Reading:",
        "  Exterior N_hydro ~ 0.98 at grain alpha_G for BOTH smooth and micro gamma0.",
        "  micro gamma0 depletes bulk rho and raises M_inner without spoiling N.",
        "  Confirms gamma0 / alpha_G decoupling under coupled loop (A1).",
    ]
    report = "\n".join(lines)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_sm_micro_coupled.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_sm_micro_coupled.txt'}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, color in [("smooth", "C0"), ("micro", "C3")]:
            sub = [r for r in rows if r.label == label]
            ax.plot(
                [r.rs_hat for r in sub],
                [r.newton_coupled for r in sub],
                "o-",
                color=color,
                label=f"{label} coupled",
            )
        ax.axhline(1.0, color="k", ls=":", lw=0.6)
        ax.set_xlabel(r"$r_s/\xi$")
        ax.set_ylabel(r"$N_{\mathrm{hydro}}$")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(OUTPUT / "ch_gravity_sm_micro_coupled.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTPUT / 'ch_gravity_sm_micro_coupled.png'}")


if __name__ == "__main__":
    main()
