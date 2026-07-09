"""
Defect microphysics — derive γ₀ from vacancy coupling (Route G1, parallel to Paper 2 κ).

Vacancy interaction (dimensionless GP units):
    L_vac = -∫ λ_v γ(x) |ψ|² d³x   →   sink term γ(x)ψ with γ₀ = 2 λ_v

Grain identification (laptop ansatz, Appendix app:lgrav):
    λ_v = m_grain c_s² / (2 E_ξ ρ_in (σ ξ)³)
    γ₀^ref = 2 λ_v

Predicts M_inner from the inner solve without Newton fitting.

  python ch_defect_microphysics.py
  python ch_mass_closure_link.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ch_dispersion_core import CHParams, G_MEAS, C
from ch_gpe_gravity import (
    calibrate_g0_matched,
    extend_schwarzschild_tail,
    hydro_newton_at_alpha_ref,
    mass_deficit_from_profile,
    mass_from_rs_hat,
    solve_gpe_spherical_sm_inner,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class DefectMicrophysicsResult:
    xi_m: float
    sigma_hat: float
    rs_hat: float
    lambda_v: float
    gamma0_ref: float
    gamma0_smooth: float
    m_grain_kg: float
    e_xi_j: float
    grain_cell_mass_kg: float
    m_inner_predicted_kg: float
    m_inner_smooth_kg: float
    m_grav_kg: float
    newton_at_ref_predicted: float
    newton_at_ref_smooth: float
    m_grav_over_m_inner_pred: float
    rs_over_sigma_xi: float


def lambda_v_vacancy_coupling(
    ch: CHParams,
    sigma_hat: float,
) -> float:
    """
    λ_v from grain vacancy cost in a sink cell of size (σ ξ)³.

    Parallel to κ = 2 λ_em / c² in Paper 2.
    """
    cell_volume = (sigma_hat * ch.xi) ** 3
    grain_mass_cell = ch.rho_in * cell_volume
    if grain_mass_cell <= 0:
        return float("nan")
    return ch.m_grain * ch.c_s**2 / (2.0 * ch.e_xi_j * grain_mass_cell)


def gamma0_from_microphysics(ch: CHParams, sigma_hat: float) -> float:
    """γ₀^ref = 2 λ_v (dimensionless sink strength)."""
    lam = lambda_v_vacancy_coupling(ch, sigma_hat)
    return 2.0 * lam


def inner_mass_for_gamma0(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    r_s_hat: float,
    r_join_hat: float = 12.0,
) -> float:
    rho_join = max((1.0 - r_s_hat / r_join_hat) ** 2, 1e-6)
    r_hat, psi, _ = solve_gpe_spherical_sm_inner(
        g0, sigma_hat, r_join_hat, rho_join, n_iter=8000
    )
    return mass_deficit_from_profile(ch, r_hat * ch.xi, np.abs(psi) ** 2)


def solve_v3_with_gamma0(
    ch: CHParams,
    g0: float,
    r_s_hat: float,
    sigma_hat: float = 0.6,
    r_join_hat: float = 12.0,
    r_max_hat: float = 300.0,
):
    """v3 profile with explicit γ₀ (not calibrate_g0_matched)."""
    from ch_gpe_gravity import _build_gravity_result

    mass_kg = mass_from_rs_hat(ch, r_s_hat)
    rho_join = max((1.0 - r_s_hat / r_join_hat) ** 2, 1e-6)
    r_inner, psi_inner, mu = solve_gpe_spherical_sm_inner(
        g0, sigma_hat, r_join_hat, rho_join
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
        solver="v3_sm_microphysics",
        s0=g0,
        sigma_hat=sigma_hat,
        fit_fraction_lo=0.12,
        fit_fraction_hi=0.5,
        r_join_hat=r_join_hat,
    )


def evaluate(
    xi_m: float = 50e-9,
    sigma_hat: float = 0.6,
    rs_hat: float = 1.0,
) -> DefectMicrophysicsResult:
    ch = CHParams(xi=xi_m, alpha_g=1.0)
    m_grav = mass_from_rs_hat(ch, rs_hat)
    g0_ref = gamma0_from_microphysics(ch, sigma_hat)
    g0_smooth, _, _, _ = calibrate_g0_matched(rs_hat, sigma_hat=sigma_hat)

    m_inner_pred = inner_mass_for_gamma0(ch, g0_ref, sigma_hat, rs_hat)
    m_inner_smooth = inner_mass_for_gamma0(ch, g0_smooth, sigma_hat, rs_hat)

    v3_pred = solve_v3_with_gamma0(ch, g0_ref, rs_hat, sigma_hat)
    v3_smooth = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)

    grain_cell = ch.rho_in * ch.xi**3
    return DefectMicrophysicsResult(
        xi_m=xi_m,
        sigma_hat=sigma_hat,
        rs_hat=rs_hat,
        lambda_v=lambda_v_vacancy_coupling(ch, sigma_hat),
        gamma0_ref=g0_ref,
        gamma0_smooth=g0_smooth,
        m_grain_kg=ch.m_grain,
        e_xi_j=ch.e_xi_j,
        grain_cell_mass_kg=grain_cell,
        m_inner_predicted_kg=m_inner_pred,
        m_inner_smooth_kg=m_inner_smooth,
        m_grav_kg=m_grav,
        newton_at_ref_predicted=hydro_newton_at_alpha_ref(v3_pred),
        newton_at_ref_smooth=hydro_newton_at_alpha_ref(v3_smooth),
        m_grav_over_m_inner_pred=m_grav / max(m_inner_pred, 1e-40),
        rs_over_sigma_xi=rs_hat / sigma_hat,
    )


def format_report(r: DefectMicrophysicsResult) -> str:
    lines = [
        "CH defect microphysics — γ₀ from vacancy coupling (Route G1)",
        f"xi = {r.xi_m:.3e} m, sigma = {r.sigma_hat}, r_s/xi = {r.rs_hat}",
        "",
        "Coupling (parallel to Paper 2 κ = 2λ/c²):",
        f"  lambda_v = {r.lambda_v:.4e}",
        f"  gamma0^ref = 2 lambda_v = {r.gamma0_ref:.4e}",
        f"  gamma0 (smooth cal) = {r.gamma0_smooth:.4f}",
        "",
        "Grain scales:",
        f"  m_grain = {r.m_grain_kg:.4e} kg",
        f"  E_xi = {r.e_xi_j:.4e} J",
        f"  rho_in * xi^3 = {r.grain_cell_mass_kg:.4e} kg",
        "",
        "Mass budgets (independent of Newton fit):",
        f"  M_grav (r_s = GM/c^2)     = {r.m_grav_kg:.4e} kg",
        f"  M_inner (gamma0^ref)      = {r.m_inner_predicted_kg:.4e} kg",
        f"  M_inner (smooth gamma0) = {r.m_inner_smooth_kg:.4e} kg",
        f"  M_grav / M_inner_pred     = {r.m_grav_over_m_inner_pred:.4e}",
        "",
        "Exterior Newton @ alpha_G^hydro,ref:",
        f"  with gamma0^ref  = {r.newton_at_ref_predicted:.4f}",
        f"  with smooth g0   = {r.newton_at_ref_smooth:.4f}",
        "",
        "Identification (Layer 2, not integral closure):",
        "  M_grav is fixed by Schwarzschild r_s; M_inner by sink microphysics.",
        f"  r_s / (sigma xi) = {r.rs_over_sigma_xi:.3f} — geometry knob for link.",
    ]
    return "\n".join(lines)


def main() -> None:
    r = evaluate()
    report = format_report(r)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_defect_microphysics.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_defect_microphysics.txt'}")


if __name__ == "__main__":
    main()
