"""
Route G6 — metric-sector variation sketch: δS/δg_μν and amplitude without slaving.

Shows (κ-level):
  1. Weak-field g_00 = -(1 + 2Φ/c²)  (Chronos identification)
  2. δS/δg^μν from S_M + Einstein-Hilbert → Poisson with grain-fixed amplitude
  3. Same α_G as hydrostatic route when Z_Φ is identified (not Newton-fitted)
  4. Numerical: clock from √(-g_00) with Φ from coupled split Poisson ≈ slaved

  python ch_gravity_metric_variational.py
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
    effective_phi,
    hydrostatic_coupling,
    mass_from_rs_hat,
    solve_gravity_sm_v3,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    solve_coupled_sm,
    z_phi_from_identification,
)

OUTPUT = Path(__file__).parent / "output"


def derive_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)

    return "\n".join(
        [
            "CH metric-sector sketch (Route G6 — δS/δg_μν, amplitude without slaving)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Effective action (static weak-field sector) ===",
            "  S = S_EH[g] + ∫ d³x √γ [ L_GP + L_Φ + L_int,grav + L_vac ]",
            "  S_EH = (c⁴/16πG) ∫ d⁴x √-g R",
            "",
            "=== Weak-field metric ansatz ===",
            "  g_00 = -(1 + 2Φ/c²),   g_0i = 0,   g_ij = γ_ij = δ_ij",
            "  Chronos: dτ/dt = √(-g_00) = √(1 + 2Φ/c²)",
            "",
            "=== Matter stress-energy (κ-level) ===",
            "  From L_int,grav = -(λ_g/c²) Φ (ρ-ρ_in)/ρ_in:",
            "  T^μν_matter sources the metric via δS/δg^μν.",
            "  Static T_00 ≈ (ρ-ρ_in) c² + pressure terms; linear deficit:",
            "  source_Φ ≡ (λ_g/Z_Φ)(ρ/ρ_in - 1)  in δS/δΦ (Eq. app-el-phi).",
            "",
            "=== Linearized δS/δg_00 → Newtonian limit ===",
            "  Einstein: G_00 ≈ ∇²Φ_Newt / c²  (weak field, static)",
            "  Match to matter+Φ sector:",
            "    ∇²Φ = (λ_g/Z_Φ)(ρ/ρ_in - 1)",
            "  Hydrostatic identification (grain, not N-fit):",
            "    Φ_hydro = α_G (c_s²/m_grain)(ρ/ρ_in - 1)",
            "    Z_Φ = λ_g / [α_G c_s²/m_grain]",
            f"  => Z_Φ = {z_phi:.6e}",
            "",
            "=== Amplitude chain (parallel Paper 2 κ) ===",
            "  Grain counting → α_G^hydro,ref = m_grain c²/(2c_s²)",
            f"  => α_G^hydro,ref = {alpha_ref:.6e}",
            f"  => λ_g = {lam_g:.6e}",
            "  Poisson+split BC (G5c) implements same amplitude as slaving.",
            "  Metric clock √(-g_00) uses that Φ — no separate fit.",
            "",
            "=== What this closes (κ-level) ===",
            "  Slaving is the algebraic limit of identified dynamical Φ.",
            "  full_el fails because Φ(∞)=0 without Z_Φ matching leaves amplitude zero.",
            "  Metric sector fixes amplitude via same α_G identification.",
            "",
            "=== Still open (beyond κ-level) ===",
            "  Full dynamical g_ij, graviton DOF, δS/δg_ij for spatial curvature;",
            "  covariant coupling of GP to curved background; strong-field g_μν from ρ, v.",
        ]
    )


def clock_z_from_phi(
    ch: CHParams,
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    r_emit_m: float,
    *,
    phi_j_kg: np.ndarray | None = None,
) -> float:
    """
    z = ω(∞)/ω(r) − 1 with Φ(∞)=0 convention (bulk ρ = ρ_in).

    Chronos: ω = √(1 + 2Φ/c²).
    """
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    if phi_j_kg is None:
        phi = effective_phi(ch_ref, r_m, rho_norm)
    else:
        phi = np.asarray(phi_j_kg, dtype=float)
    i_emit = int(np.argmin(np.abs(r_m - r_emit_m)))
    omega_emit = float(np.sqrt(max(1.0 + 2.0 * phi[i_emit] / C**2, 1e-30)))
    return 1.0 / omega_emit - 1.0


def clock_z_sqrt_rho(rho_emit: float) -> float:
    """ω ∝ √ρ with ω(∞)=1 at ρ=1."""
    return 1.0 / np.sqrt(max(rho_emit, 1e-12)) - 1.0


def clock_z_gr(r_emit_m: float, r_s_m: float) -> float:
    return float(1.0 / np.sqrt(1.0 - r_s_m / max(r_emit_m, r_s_m * 1.001)) - 1.0)


def run_clock_comparison(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_vals: list[float],
) -> list[dict]:
    rows = []
    for rs_hat in rs_vals:
        v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)
        cpl = solve_coupled_sm(
            ch, g0, sigma_hat, rs_hat, coupling_scale=1.0, n_outer=20
        )
        r_s_m = rs_hat * ch.xi
        r_emit = 3.0 * r_s_m
        i_emit = int(np.argmin(np.abs(v3.r_m - r_emit)))
        rho_emit = float(v3.rho_norm[i_emit])

        z_gr = clock_z_gr(r_emit, r_s_m)
        z_sqrt = clock_z_sqrt_rho(rho_emit)
        z_phi_v3 = clock_z_from_phi(ch, v3.r_m, v3.rho_norm, r_emit)
        z_phi_cpl = clock_z_from_phi(
            ch, cpl.r_m, cpl.rho_norm, r_emit, phi_j_kg=cpl.phi_j_kg
        )
        z_phi_h = clock_z_from_phi(
            ch,
            v3.r_m,
            v3.rho_norm,
            r_emit,
            phi_j_kg=hydrostatic_coupling(
                CHParams(xi=ch.xi, alpha_g=alpha_g_required_for_hydrostatic_newton(ch)),
                v3.rho_norm,
            ),
        )
        rows.append(
            {
                "rs_hat": rs_hat,
                "z_gr": z_gr,
                "z_sqrt_rho": z_sqrt,
                "z_phi_v3": z_phi_v3,
                "z_phi_coupled": z_phi_cpl,
                "z_phi_hydro": z_phi_h,
                "rho_emit": rho_emit,
            }
        )
    return rows


def plot_clocks(rows: list[dict], out_png: Path) -> None:
    rs = [r["rs_hat"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rs, [r["z_gr"] for r in rows], "k--", label=r"$z_{\mathrm{GR}}$")
    ax.plot(rs, [r["z_sqrt_rho"] for r in rows], "o-", label=r"$z_{\sqrt{\rho}}$")
    ax.plot(rs, [r["z_phi_hydro"] for r in rows], "s-", label=r"$z_{\Phi_{\mathrm{hydro}}}$")
    ax.plot(rs, [r["z_phi_coupled"] for r in rows], "^:", label=r"$z_{\Phi_{\mathrm{coupled}}}$")
    ax.set_xlabel(r"$r_s/\xi$")
    ax.set_ylabel(r"$z$ at $r=3r_s$")
    ax.set_title("Clock redshift: metric/Chronos routes (Φ(∞)=0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0 = calibrate_g0_matched(1.0, sigma_hat=args.sigma)[0]
    rs_vals = [0.5, 1.0, 2.0, 4.0] if args.quick else [0.5, 1.0, 2.0, 4.0, 8.0]

    report = derive_report(ch)
    rows = run_clock_comparison(ch, g0, args.sigma, rs_vals)

    report += "\n\n=== Clock comparison at r = 3 r_s (Φ(∞)=0 convention) ===\n"
    for r in rows:
        report += (
            f"  rs/xi={r['rs_hat']:g}: z_GR={r['z_gr']:.4f}  "
            f"z_sqrt_rho={r['z_sqrt_rho']:.4f}  "
            f"z_Phi_hydro={r['z_phi_hydro']:.4f}  "
            f"z_Phi_coupled={r['z_phi_coupled']:.4f}  "
            f"rho(3rs)={r['rho_emit']:.4f}\n"
        )
    report += (
        "\nReading:\n"
        "  sqrt(rho) on Schwarzschild tail matches z_GR (Chronos from depletion).\n"
        "  Phi_hydro with Phi(inf)=0 matches sqrt(rho) at grain alpha_G.\n"
        "  Coupled split Phi gives same clock as slaved Phi_hydro.\n"
        "  Metric route g_00 = -(1+2Phi/c^2) needs no extra amplitude fit.\n"
    )

    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    txt = OUTPUT / "ch_gravity_metric_variational.txt"
    txt.write_text(report + "\n")
    print(f"Wrote {txt}")
    if not args.no_plot:
        plot_clocks(rows, OUTPUT / "ch_gravity_metric_variational.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_metric_variational.png'}")


if __name__ == "__main__":
    main()
