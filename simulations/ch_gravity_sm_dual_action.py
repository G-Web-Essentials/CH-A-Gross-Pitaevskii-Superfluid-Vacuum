"""
Route G9c — dual-Φ sector in S_M + S_EH (field-theoretic κ-level).

Extended static action:
  S = S_EH[g; Φ_g] + ∫ [ L_GP + L_Φ(Φ_dyn) + L_int,grav(Φ_dyn) + L_vac ]

  Φ_dyn: EL field from δS/δΦ  → GP + Poisson (force, Z_Φ ID)
  Φ_g:   enters g_00 = −(1 + 2Φ_g/c²)  (chronos, Z_g ID)
  Φ_g = Φ_dyn / Z_g  on exterior  [bridge, from action]
  Φ_g = −GM/(2r)     on exterior  [dual overlay, Schwarzschild BC]

Validates split / dual / bridge poisson_mode in coupled loop.

  python ch_gravity_sm_dual_action.py
  python ch_gravity_sm_dual_action.py --quick
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from ch_acoustic_light_bending import deflection_gr_full
from ch_dispersion_core import C, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    hydrostatic_newton_factor,
    mass_from_rs_hat,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    solve_coupled_sm,
    z_g_from_identification,
    z_phi_from_identification,
    z_clock_from_phi,
    z_clock_gr,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class DualActionPoint:
    rs_hat: float
    mode: str
    n_dyn: float
    z_dyn_over_gr: float
    z_g_over_gr: float
    alpha_g_over_full: float


def deflection_null_weak(b_m: float, r_m: np.ndarray, phi: np.ndarray) -> float:
    r = np.asarray(r_m, dtype=float)
    phi = np.asarray(phi, dtype=float)
    order = np.argsort(r)
    r, phi = r[order], phi[order]
    dphi_dr = np.gradient(phi, r)

    def dphi_dr_at(rq: float) -> float:
        return float(np.interp(rq, r, dphi_dr))

    b = float(b_m)

    def integrand(z: float) -> float:
        rv = float(np.hypot(b, z))
        if rv < r[0]:
            return 0.0
        return (4.0 / C**2) * dphi_dr_at(rv) * (b / rv)

    alpha, _ = quad(integrand, -200.0 * b, 200.0 * b, limit=200)
    return float(abs(alpha))


def action_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    z_g = z_g_from_identification(ch, alpha_ref)
    hnewt = hydrostatic_newton_factor(CHParams(xi=ch.xi, alpha_g=alpha_ref))

    return "\n".join(
        [
            "CH dual-Φ action sector (Route G9c — κ-level)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Extended static action ===",
            "  S = S_EH[g; Φ_g] + ∫ d³x [ L_GP + (Z_Φ/2c²)|∇Φ|² + L_int,grav(Φ) + L_vac ]",
            "",
            "  ONE EL field Φ ≡ Φ_dyn sourced by δS/δΦ (Eq. app-el-phi).",
            "  GP + Poisson use Φ_dyn — force channel, Z_Φ identification.",
            "",
            "  Metric sector (δS/δg_00, weak field):",
            "    g_00 = −(1 + 2Φ_g/c²)",
            "    Φ_g = Φ_dyn / Z_g   on exterior  [algebraic metric ID]",
            "  Equivalently: g_00 = −(1 + 2Φ_dyn/(Z_g c²)).",
            "",
            "=== Identifications (grain-fixed, not N/z fit) ===",
            f"  Z_Φ = λ_g / [α_G c_s²/m] = {z_phi:.6e}  [force/Poisson]",
            f"  Z_g = 4 α_G c_s²/(m c²) = {z_g:.6f}  [metric/chronos]",
            f"  hydrostatic_newton_factor = {hnewt:.6f}  (= Z_g/2 at grain α_G)",
            "",
            "=== Implementation in coupled loop ===",
            "  poisson_mode=split:   Φ_g = Φ_dyn  (baseline; z overshoots)",
            "  poisson_mode=bridge: Φ_dyn loop; Φ_g = Φ_dyn/Z_g exterior",
            "  poisson_mode=dual:    Φ_dyn loop; Φ_g = −GM/(2r) exterior (BC)",
            "",
            "=== κ-level closure ===",
            "  G9 factor-of-2 split = two δS identifications (Z_Φ vs Z_g).",
            "  Not two independent fit parameters.",
            "  bridge implements action ID; dual implements Schwarzschild BC.",
        ]
    )


def evaluate_mode(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
    mode: str,
) -> DualActionPoint:
    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        n_outer=20,
        coupling_scale=1.0,
        poisson_mode=mode,
    )
    r_emit = 20.0 * ch.xi
    b_m = 10.0 * rs_hat * ch.xi
    mass_kg = mass_from_rs_hat(ch, rs_hat)
    alpha_full = float(deflection_gr_full(mass_kg, b_m))
    i_emit = int(np.argmin(np.abs(cpl.r_m - r_emit)))
    z_gr = z_clock_gr(r_emit, rs_hat * ch.xi)
    z_dyn = z_clock_from_phi(float(cpl.phi_j_kg[i_emit]))

    if cpl.phi_g_j_kg is not None:
        z_g = z_clock_from_phi(float(cpl.phi_g_j_kg[i_emit]))
        phi_bend = cpl.phi_g_j_kg
    else:
        z_g = z_dyn
        phi_bend = cpl.phi_j_kg

    alpha = deflection_null_weak(b_m, cpl.r_m, phi_bend)

    return DualActionPoint(
        rs_hat=rs_hat,
        mode=mode,
        n_dyn=cpl.newton_coupled,
        z_dyn_over_gr=z_dyn / max(z_gr, 1e-30),
        z_g_over_gr=z_g / max(z_gr, 1e-30),
        alpha_g_over_full=alpha / max(alpha_full, 1e-30),
    )


def format_report(rows: list[DualActionPoint]) -> str:
    rs_vals = sorted({r.rs_hat for r in rows})
    lines = ["", "=== Coupled loop: split vs bridge vs dual (clock at 20ξ) ==="]
    for rs in rs_vals:
        sub = [r for r in rows if r.rs_hat == rs]
        lines.append(f"  r_s/xi = {rs:g}")
        for r in sorted(sub, key=lambda x: x.mode):
            lines.append(
                f"    {r.mode:6s}: N={r.n_dyn:.4f}  "
                f"z_dyn/z_GR={r.z_dyn_over_gr:.3f}  z_g/z_GR={r.z_g_over_gr:.3f}  "
                f"α_g/α_full={r.alpha_g_over_full:.3f}"
            )
    lines += [
        "",
        "Reading:",
        "  split: N≈1, z_dyn≈2× GR (using Φ_dyn in g_00 — pre-G9b error).",
        "  bridge: N≈1, z_g≈0.88–0.97 via Φ_g=Φ_dyn/Z_g (action ID; linear tail).",
        "  dual: N≈1, z_g≈1 via −GM/(2r) BC (nonlinear Schwarzschild tail).",
    ]
    return "\n".join(lines)


def plot_results(rows: list[DualActionPoint], out_png: Path) -> None:
    modes = ["split", "bridge", "dual"]
    colors = {"split": "C0", "bridge": "C2", "dual": "C3"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for mode in modes:
        sub = sorted([r for r in rows if r.mode == mode], key=lambda r: r.rs_hat)
        if not sub:
            continue
        rs = [r.rs_hat for r in sub]
        axes[0].plot(rs, [r.n_dyn for r in sub], "o-", color=colors[mode], label=mode)
        axes[1].plot(
            rs, [r.z_g_over_gr for r in sub], "o-", color=colors[mode], label=mode
        )
        axes[2].plot(
            rs, [r.alpha_g_over_full for r in sub], "o-", color=colors[mode], label=mode
        )
    for ax in axes:
        ax.axhline(1.0, color="k", ls=":", lw=0.6)
        ax.set_xlabel(r"$r_s/\xi$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$N_{\mathrm{hydro}}$")
    axes[0].set_title(r"$\Phi_{\mathrm{dyn}}$ force")
    axes[1].set_ylabel(r"$z_g/z_{\mathrm{GR}}$")
    axes[1].set_title(r"$\Phi_g$ clocks at $20\xi$")
    axes[2].set_ylabel(r"$\alpha/\alpha_{\mathrm{full}}$")
    axes[2].set_title(r"$\Phi_g$ bending")
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
    rs_vals = [1.0, 2.0, 4.0, 8.0] if args.quick else [0.5, 1.0, 2.0, 4.0, 8.0]
    modes = ("split", "bridge", "dual")

    report = action_report(ch)
    rows = [
        evaluate_mode(ch, g0, args.sigma, rs, mode)
        for rs in rs_vals
        for mode in modes
    ]
    report += format_report(rows)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_sm_dual_action.txt").write_text(report + "\n")
    print(f"Wrote {OUTPUT / 'ch_gravity_sm_dual_action.txt'}")
    if not args.no_plot:
        plot_results(rows, OUTPUT / "ch_gravity_sm_dual_action.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_sm_dual_action.png'}")


if __name__ == "__main__":
    main()
