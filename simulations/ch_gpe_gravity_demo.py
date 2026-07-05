#!/usr/bin/env python3
"""
CH Gravity demo — v2 (V_def + tail patch) and v3 (S_M source, no patch).

  python ch_gpe_gravity_demo.py
  python ch_gpe_gravity_demo.py --v3-only
  python ch_gpe_gravity_demo.py --analytic-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    analyze_analytic_profile,
    calibrate_alpha_g_from_defect,
    format_alpha_g_calibration_lines,
    hydrostatic_alpha_g,
    mass_from_rs_hat,
    solve_gravity_defect,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def plot_result(result, title: str, out_path: Path) -> None:
    r = result.r_m
    m = result.mass_kg
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].semilogx(r, result.rho_norm, "b-", lw=1.5)
    axes[0, 0].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[0, 0].set_xlabel(r"$r$ [m]")
    axes[0, 0].set_ylabel(r"$\rho/\rho_{\mathrm{in}}$")
    axes[0, 0].set_title("Density (no tail patch)" if result.solver == "v3_sm" else "Density")
    axes[0, 0].grid(True, alpha=0.3)

    a_newton = G_MEAS * m / np.clip(r**2, 1e-60, None)
    axes[0, 1].loglog(r, np.abs(result.accel_m_s2), "r-", label=r"$|a|$ total")
    if result.phi_q is not None:
        accel_q = -np.gradient(result.phi_q, r)
        accel_h = -np.gradient(result.phi_hydro, r)
        axes[0, 1].loglog(r, np.abs(accel_q), "c--", alpha=0.7, label=r"$|a|$ from $Q$")
        axes[0, 1].loglog(r, np.abs(accel_h), "m--", alpha=0.7, label=r"$|a|$ hydro")
    axes[0, 1].loglog(r, a_newton, "k--", alpha=0.5, label=r"$GM/r^2$")
    axes[0, 1].axvspan(result.r_fit_lo_m, result.r_fit_hi_m, alpha=0.12, color="green")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_title("Acceleration channels")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogx(r, result.g_eff, "m-", lw=1.5)
    axes[1, 0].axhline(G_MEAS, color="k", ls="--", label=r"$G_{\mathrm{meas}}$")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title(r"$G_{\mathrm{eff}}(r)=|a|r^2/M$")
    axes[1, 0].grid(True, alpha=0.3)

    lines = [
        f"{title}",
        f"solver: {result.solver}",
        f"M={m:.3e} kg, r_s/ξ={result.r_s_schwarzschild_m/result.xi_m:.2f}",
        f"Newton total = {result.newton_slope:.3e}",
    ]
    if result.newton_slope_q is not None:
        lines.append(f"Newton Q     = {result.newton_slope_q:.3e}")
        lines.append(f"Newton hydro = {result.newton_slope_hydro:.3e}")
    if result.s0 is not None:
        lines.append(f"s₀={result.s0:.3f}, σ={result.sigma_hat:.2f} ξ")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_report(label: str, result) -> None:
    print(f"\n--- {label} [{result.solver}] ---")
    print(f"  M target / deficit: {result.mass_kg:.4e} / {result.mass_deficit_kg:.4e} kg")
    print(f"  r_s = {result.r_s_schwarzschild_m:.4e} m  (r_s/ξ = {result.r_s_schwarzschild_m/result.xi_m:.3f})")
    if result.s0 is not None:
        print(f"  s₀ = {result.s0:.4f},  σ = {result.sigma_hat:.2f} ξ")
    if result.v0 > 0:
        print(f"  V₀ = {result.v0:.4f},  R_c = {result.rc_hat:.2f} ξ")
    print(f"  Fit annulus: [{result.r_fit_lo_m:.3e}, {result.r_fit_hi_m:.3e}] m")
    print(f"  Newton total |a|/(GM/r²) = {result.newton_slope:.4e}")
    if result.newton_slope_q is not None:
        print(f"  Newton Q only            = {result.newton_slope_q:.4e}")
        print(f"  Newton hydro only        = {result.newton_slope_hydro:.4e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CH gravity v2/v3 spherical GPE demo")
    parser.add_argument("--mass", type=float, default=None, help="Defect mass [kg]")
    parser.add_argument("--rs-hat", type=float, default=1.0, help="Schwarzschild radius in units of ξ")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--analytic-only", action="store_true", help="Skip numerical GPE")
    parser.add_argument("--v3-only", action="store_true", help="Run only S_M v3 solver")
    parser.add_argument("--skip-v2", action="store_true", help="Skip v2 V_def solver")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi, alpha_g=1.0)
    mass = args.mass if args.mass is not None else mass_from_rs_hat(ch, args.rs_hat)

    print("=" * 72)
    print("CH GRAVITY — v3 S_M source (no tail patch) + v2 comparison")
    print("=" * 72)
    print(f"ξ={ch.xi:.3e} m, ρ_in={ch.rho_in:.3e} kg/m³")
    print(f"M={mass:.4e} kg, r_s/ξ={G_MEAS*mass/C**2/ch.xi:.3f}")
    print(f"Hydrostatic Newton factor (α_G=1) = {hydrostatic_alpha_g(ch):.4e}")
    print(f"α_G required (hydro only)         = {alpha_g_required_for_hydrostatic_newton(ch):.4e}")

    ana = analyze_analytic_profile(ch, mass)
    print_report("Analytic (1−r_s/r)² hydrostatic", ana)
    plot_result(ana, "Analytic depletion", OUTPUT / "ch_gpe_gravity_analytic.png")

    results: list[tuple[str, object]] = [("analytic", ana)]

    if not args.analytic_only:
        if not args.v3_only and not args.skip_v2:
            print("\nSolving v2 (V_def + Schwarzschild tail patch)...")
            v2 = solve_gravity_defect(ch, r_s_hat=args.rs_hat)
            print_report("v2 V_def + tail", v2)
            plot_result(v2, "Gravity v2", OUTPUT / "ch_gpe_gravity_v2.png")
            results.append(("v2", v2))

        print("\nSolving v3 (S_M source, no tail patch)...")
        v3 = solve_gravity_sm_v3(ch, r_s_hat=args.rs_hat)
        print_report("v3 S_M only", v3)
        cal_v3, v3_cal = calibrate_alpha_g_from_defect(v3)
        print("  α_G calibration (Newton → 1):")
        for line in format_alpha_g_calibration_lines(cal_v3):
            print(line)
        plot_result(v3_cal, "Gravity v3 — S_M (α_G calibrated)", OUTPUT / "ch_gpe_gravity_v3.png")
        results.append(("v3", v3_cal))

    summary = OUTPUT / "ch_gpe_gravity_report.txt"
    with summary.open("w") as f:
        f.write("CH gravity report\n\n")
        for label, res in results:
            f.write(f"[{label}] solver={res.solver}\n")
            f.write(f"  newton_total={res.newton_slope}\n")
            if res.newton_slope_q is not None:
                f.write(f"  newton_q={res.newton_slope_q}\n")
                f.write(f"  newton_hydro={res.newton_slope_hydro}\n")
            f.write(f"  mass_deficit_kg={res.mass_deficit_kg}\n\n")

    print(f"\nWrote {OUTPUT / 'ch_gpe_gravity_analytic.png'}")
    if not args.analytic_only:
        if not args.v3_only and not args.skip_v2:
            print(f"Wrote {OUTPUT / 'ch_gpe_gravity_v2.png'}")
        print(f"Wrote {OUTPUT / 'ch_gpe_gravity_v3.png'}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
