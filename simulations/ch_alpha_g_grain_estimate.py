"""
Grain-scale reference for hydrostatic Newton coupling alpha_G.

The hydrostatic channel uses Phi_hydro = alpha_G (c_s^2/m_grain) (rho/rho_in - 1).
Linear far-field matching (Schwarzschild-depletion ansatz) gives the grain reference

  alpha_G^hydro,ref = m_grain c^2 / (2 c_s^2)   (= m_grain/2 when c_s = c)

At alpha_G = 1 the same channel overshoots Newton; alpha_G^cal fits the numerical
v3 GP profile post hoc. This script compares all three on the laptop.

  python ch_alpha_g_grain_estimate.py
  python ch_alpha_g_grain_estimate.py --xi 50e-9 --rs-hat 1.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ch_gpe_gravity import (
    CHParams,
    alpha_g_profile_corrected,
    alpha_g_required_for_hydrostatic_newton,
    analyze_analytic_profile,
    calibrate_alpha_g_from_defect,
    hydrostatic_newton_factor,
    mass_from_rs_hat,
    recalibrate_result_with_alpha_g,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class AlphaGReport:
    xi_m: float
    m_grain_kg: float
    rho_in: float
    alpha_g_assumed: float
    alpha_g_hydro_ref: float
    f_hydro_at_assumed: float
    newton_analytic_at_assumed: float
    newton_analytic_at_hydro_ref: float
    newton_v3_at_assumed: float
    newton_v3_at_hydro_ref: float
    alpha_g_calibrated: float
    newton_v3_after_cal: float
    alpha_g_profile: float
    f_profile: float
    newton_v3_after_profile: float
    ratio_cal_over_hydro_ref: float


def build_report(xi_m: float, rs_hat: float = 1.0) -> AlphaGReport:
    ch1 = CHParams(xi=xi_m, alpha_g=1.0)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch1)
    ch_ref = CHParams(xi=xi_m, alpha_g=alpha_ref)
    mass = mass_from_rs_hat(ch1, rs_hat)

    ana1 = analyze_analytic_profile(ch1, mass)
    ana_ref = analyze_analytic_profile(ch_ref, mass)

    v3_1 = solve_gravity_sm_v3(ch1, r_s_hat=rs_hat)
    cal, v3_cal = calibrate_alpha_g_from_defect(v3_1)
    v3_ref = recalibrate_result_with_alpha_g(v3_1, ch_ref)
    alpha_prof, f_prof, v3_prof = alpha_g_profile_corrected(v3_1)

    ratio = (
        cal.alpha_g_calibrated / alpha_ref
        if alpha_ref > 0 and cal.alpha_g_calibrated == cal.alpha_g_calibrated
        else float("nan")
    )

    return AlphaGReport(
        xi_m=xi_m,
        m_grain_kg=ch1.m_grain,
        rho_in=ch1.rho_in,
        alpha_g_assumed=1.0,
        alpha_g_hydro_ref=alpha_ref,
        f_hydro_at_assumed=hydrostatic_newton_factor(ch1),
        newton_analytic_at_assumed=float(ana1.newton_slope),
        newton_analytic_at_hydro_ref=float(ana_ref.newton_slope),
        newton_v3_at_assumed=float(v3_1.newton_slope),
        newton_v3_at_hydro_ref=float(v3_ref.newton_slope),
        alpha_g_calibrated=float(cal.alpha_g_calibrated),
        newton_v3_after_cal=float(v3_cal.newton_slope),
        alpha_g_profile=float(alpha_prof),
        f_profile=float(f_prof),
        newton_v3_after_profile=float(v3_prof.newton_slope),
        ratio_cal_over_hydro_ref=float(ratio),
    )


def format_report(r: AlphaGReport) -> str:
    lines = [
        "CH alpha_G grain reference vs calibration",
        f"xi = {r.xi_m:.3e} m",
        f"m_grain = {r.m_grain_kg:.4e} kg",
        f"rho_in = {r.rho_in:.4e} kg/m^3",
        "",
        "Grain hydrostatic reference (linear / analytic matching):",
        f"  alpha_G^hydro,ref = m_grain c^2 / (2 c_s^2) = {r.alpha_g_hydro_ref:.4e}",
        f"  F_hydro at alpha_G=1 = {r.f_hydro_at_assumed:.4e}",
        "",
        "Analytic Schwarzschild-depletion profile:",
        f"  Newton factor @ alpha_G=1          = {r.newton_analytic_at_assumed:.4e}",
        f"  Newton factor @ alpha_G^hydro,ref  = {r.newton_analytic_at_hydro_ref:.4f}",
        "",
        "Numerical v3 GP defect (S_M, matched exterior):",
        f"  Newton factor @ alpha_G=1          = {r.newton_v3_at_assumed:.4e}",
        f"  Newton factor @ alpha_G^hydro,ref  = {r.newton_v3_at_hydro_ref:.4f}",
        f"  alpha_G^cal (slope fit)            = {r.alpha_g_calibrated:.4e}",
        f"  Newton factor after slope cal      = {r.newton_v3_after_cal:.4f}",
        f"  f_profile = 1/N_hydro@ref          = {r.f_profile:.4e}",
        f"  alpha_G profile rule               = {r.alpha_g_profile:.4e}",
        f"  Newton factor after profile rule   = {r.newton_v3_after_profile:.6f}",
        f"  alpha_G^cal / alpha_G^hydro,ref    = {r.ratio_cal_over_hydro_ref:.4e}",
        "",
        "Interpretation:",
        "  alpha_G^hydro,ref closes Newton on the analytic ansatz without fitting.",
        "  Primary: alpha_G^hydro,ref closes exterior Newton (~0.98 v3 & analytic).",
        "  Optional f = 1/N_hydro@ref is ~2% trim (earlier f~0.004 was full-grid artefact).",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="alpha_G grain reference vs v3 calibration")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--rs-hat", type=float, default=1.0, help="Schwarzschild radius in units of xi")
    args = parser.parse_args()

    report = build_report(args.xi, rs_hat=args.rs_hat)
    text = format_report(report)
    print(text)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "ch_alpha_g_grain_estimate.txt"
    out_path.write_text(text + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
