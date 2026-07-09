"""
Route G9b+ — derive Z_g ≈ 2 from action identification (not a fit knob).

Two κ-level identifications from δS:
  Z_Φ  — Poisson δS/δΦ matches Φ_hydro (force channel, G5/G6)
  Z_g  — g_00 = −(1+2Φ_g/c²) matches GR with repo r_s = GM/c²

At grain α_G^hydro,ref = m_grain c²/(2c_s²):
  Z_g = 4 α_G c_s² / (m_grain c²) = 2.

Validates numerically on Schwarzschild tail and coupled dual mode.

  python ch_gravity_zg_identification.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    hydrostatic_coupling,
    hydrostatic_newton_factor,
    mass_from_rs_hat,
    schwarzschild_depletion_profile,
)
from ch_gravity_sm_coupled import (
    phi_metric_exterior,
    solve_coupled_sm,
    z_g_from_identification,
)

OUTPUT = Path(__file__).parent / "output"


def derive_zg_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    hnewt = hydrostatic_newton_factor(ch_ref)
    zg = z_g_from_identification(ch, alpha_ref)
    zg_direct = 4.0 * alpha_ref * ch.c_s**2 / (ch.m_grain * C**2)

    return "\n".join(
        [
            "CH metric amplitude bridge Z_g (G9b+ — from identification)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Two roles, two identifications ===",
            "  Φ_dyn from δS/δΦ:  ∇²Φ = (λ_g/Z_Φ)(ρ/ρ_in − 1)",
            "    Z_Φ matches linear Φ_hydro = α_G(c_s²/m)(ρ/ρ_in − 1)  [force]",
            "  Φ_g enters g_00:      g_00 = −(1 + 2Φ_g/c²)",
            "    GR weak field (repo r_s = GM/c²):  g_00 = −(1 − r_s/r)",
            "    => Φ_g = −GM/(2r) on exterior tail",
            "",
            "=== Tail linearization (ρ = (1−r_s/r)²) ===",
            "  Φ_hydro ≈ −2 α_G (c_s²/m) r_s/r",
            "  Φ_g     = −r_s c² / (2r)",
            "  Ratio |Φ_hydro|/|Φ_g| = 4 α_G c_s² / (m c²)",
            "",
            "=== Grain reference ===",
            f"  α_G^hydro,ref = m_grain c²/(2c_s²) = {alpha_ref:.6e}",
            f"  hydrostatic_newton_factor = 2 α_G c_s²/(m c²) = {hnewt:.6f}",
            f"  Z_g = 4 α_G c_s²/(m c²) = 2 × hydrostatic_newton_factor = {zg:.6f}",
            f"  (direct) Z_g = {zg_direct:.6f}",
            "",
            "=== κ-level prescription (G9b dual) ===",
            "  Coupled loop uses Φ_dyn (split/hydro) for GP + N_hydro.",
            "  Metric sector:  Φ_g = Φ_dyn / Z_g  on exterior, or explicit −GM/(2r).",
            "  Z_g = 2 at grain α_G — convention bridge from r_s = GM/c², not tuned to data.",
            "",
            "=== What this closes ===",
            "  G9 factor-of-2 split explained by two δS identifications.",
            "  dual poisson_mode in G5c implements Φ_g overlay post-loop.",
        ]
    )


def tail_ratio_check(ch: CHParams, rs_hat: float) -> dict[str, float]:
    """|Φ_hydro|/|Φ_g| on analytic Schwarzschild tail."""
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    r_s_m = rs_hat * ch.xi
    r_m = np.geomspace(r_s_m * 3.0, 50.0 * ch.xi, 200)
    rho = schwarzschild_depletion_profile(ch, mass_kg, r_m)
    phi_h = hydrostatic_coupling(ch_ref, rho)
    phi_g = phi_metric_exterior(r_m, phi_h, mass_kg, r_join_m=12.0 * ch.xi)
    ratio = float(np.median(np.abs(phi_h) / np.clip(np.abs(phi_g), 1e-99, None)))
    zg = z_g_from_identification(ch, alpha_ref)
    return {
        "rs_hat": rs_hat,
        "median_phi_ratio": ratio,
        "z_g_identified": zg,
        "ratio_minus_zg": ratio - zg,
    }


def dual_coupled_check(
    ch: CHParams,
    rs_hat: float,
    sigma_hat: float,
    g0: float,
) -> dict[str, float]:
    res = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        n_outer=20,
        coupling_scale=1.0,
        poisson_mode="dual",
    )
    return {
        "rs_hat": rs_hat,
        "n_hydro": res.newton_coupled,
        "z_g_20xi_over_gr": res.z_g_20xi / max(res.z_gr_20xi, 1e-30),
        "z_g_3rs_over_gr": res.z_g_3rs / max(res.z_gr_3rs, 1e-30),
        "z_dyn_20xi_over_gr": float("nan"),  # filled below
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0 = calibrate_g0_matched(1.0, sigma_hat=args.sigma)[0]
    rs_vals = [1.0, 4.0, 8.0] if args.quick else [0.5, 1.0, 2.0, 4.0, 8.0]

    report = derive_zg_report(ch)
    report += "\n\n=== Analytic tail ratio |Φ_hydro|/|Φ_g| ===\n"
    tail_rows = [tail_ratio_check(ch, rs) for rs in rs_vals]
    for row in tail_rows:
        report += (
            f"  rs/xi={row['rs_hat']:g}: median ratio={row['median_phi_ratio']:.4f}  "
            f"Z_g={row['z_g_identified']:.4f}  Δ={row['ratio_minus_zg']:+.4f}\n"
        )

    report += "\n=== Coupled dual mode (embedded G9b) ===\n"
    dual_rows = []
    for rs in rs_vals:
        res = solve_coupled_sm(
            ch,
            g0,
            args.sigma,
            rs,
            n_outer=20,
            coupling_scale=1.0,
            poisson_mode="dual",
        )
        dual_rows.append(res)
        report += (
            f"  rs/xi={rs:g}: N={res.newton_coupled:.4f}  "
            f"z_g(20ξ)/z_GR={res.z_g_20xi / max(res.z_gr_20xi, 1e-30):.3f}  "
            f"z_g(3r_s)/z_GR={res.z_g_3rs / max(res.z_gr_3rs, 1e-30):.3f}\n"
        )

    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_zg_identification.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_zg_identification.txt'}")

    if not args.no_plot:
        alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
        ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
        mass_kg = mass_from_rs_hat(ch, 4.0)
        r_m = np.geomspace(ch.xi * 2, 80.0 * ch.xi, 400)
        rho = schwarzschild_depletion_profile(ch, mass_kg, r_m, r_core_m=12.0 * ch.xi)
        phi_h = hydrostatic_coupling(ch_ref, rho)
        phi_g = phi_metric_exterior(r_m, phi_h, mass_kg, 12.0 * ch.xi)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(r_m / ch.xi, phi_h, label=r"$\Phi_{\mathrm{hydro}}$")
        axes[0].plot(r_m / ch.xi, phi_g, label=r"$\Phi_g=-GM/(2r)$")
        axes[0].plot(r_m / ch.xi, phi_h / 2.0, "k:", label=r"$\Phi_{\mathrm{hydro}}/Z_g$")
        axes[0].set_xlabel(r"$r/\xi$")
        axes[0].set_ylabel(r"$\Phi$ [J/kg]")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[0].set_title(r"Tail potentials ($r_s/\xi=4$)")

        rs_plot = [r.rs_hat for r in dual_rows]
        axes[1].plot(
            rs_plot,
            [r.z_g_20xi / max(r.z_gr_20xi, 1e-30) for r in dual_rows],
            "o-",
            label=r"$z_g/z_{\mathrm{GR}}$ at $20\xi$",
        )
        axes[1].plot(
            rs_plot,
            [r.newton_coupled for r in dual_rows],
            "s--",
            label=r"$N_{\mathrm{hydro}}$",
        )
        axes[1].axhline(1.0, color="k", ls=":", lw=0.6)
        axes[1].set_xlabel(r"$r_s/\xi$")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].set_title("dual poisson_mode in coupled loop")
        fig.tight_layout()
        fig.savefig(OUTPUT / "ch_gravity_zg_identification.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTPUT / 'ch_gravity_zg_identification.png'}")


if __name__ == "__main__":
    main()
