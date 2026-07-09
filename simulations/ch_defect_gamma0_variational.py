"""
Variational derivation of vacancy coupling λ_v and γ₀ (Route G2b).

Extends ch_defect_microphysics.py:
  1. Euler–Lagrange from L_GP + L_vac → γ₀ = 2λ_v
  2. λ_v from grain vacancy energy balance (same as grain counting, derived via δF=0)
  3. γ₀^smooth as core regularization (ρ_center target), not microphysics

  python ch_defect_gamma0_variational.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ch_defect_microphysics import (
    evaluate,
    format_report,
    gamma0_from_microphysics,
    inner_mass_for_gamma0,
    lambda_v_vacancy_coupling,
)
from ch_dispersion_core import CHParams
from ch_gpe_gravity import calibrate_g0_matched

OUTPUT = Path(__file__).parent / "output"


def lambda_v_from_energy_balance(ch: CHParams, sigma_hat: float) -> float:
    """
    Variational grain balance: removing one (σξ)³ cell costs E_ξ.

    δF/δ(occupation) = 0 for a single vacancy in a cell of mass M_cell = ρ_in (σξ)³:
      λ_v = m_grain c_s² / (2 E_ξ M_cell)
    """
    m_cell = ch.rho_in * (sigma_hat * ch.xi) ** 3
    return ch.m_grain * ch.c_s**2 / (2.0 * ch.e_xi_j * m_cell)


def gamma0_smooth_role(r_s_hat: float, sigma_hat: float = 0.6) -> dict[str, float]:
    """Document smooth γ₀ as core regularization (ρ_center target, not vacancy rate)."""
    rho_join_target = max((1.0 - r_s_hat / 12.0) ** 2, 1e-6)
    rho_center_target = max(1.0 - 0.35 * r_s_hat, 0.3)
    g0, _, _, _ = calibrate_g0_matched(r_s_hat, sigma_hat=sigma_hat)
    return {
        "gamma0_smooth": g0,
        "rho_join_bc": rho_join_target,
        "rho_center_target": rho_center_target,
    }


def main() -> None:
    xi = 50e-9
    sigma = 0.6
    rs_hat = 1.0
    ch = CHParams(xi=xi)

    lam_count = lambda_v_vacancy_coupling(ch, sigma)
    lam_var = lambda_v_from_energy_balance(ch, sigma)
    g0_micro = gamma0_from_microphysics(ch, sigma)
    smooth = gamma0_smooth_role(rs_hat, sigma)

    r = evaluate(xi_m=xi, sigma_hat=sigma, rs_hat=rs_hat)

    lines = [
        "CH defect γ₀ — variational derivation (Route G2b)",
        f"xi = {xi:.3e} m, sigma = {sigma}, r_s/xi = {rs_hat}",
        "",
        "=== Euler–Lagrange (dimensionless GP) ===",
        "  L = L_GP + L_vac,   L_vac = -λ_v ∫ γ(x)|ψ|² d³x",
        "  δL/δψ* → stationary GP includes sink term -γ_sink ψ",
        "  With γ_sink(x) = γ₀ exp(-r²/σ²ξ²) normalized at core:  γ₀ = 2λ_v",
        "",
        "=== λ_v from vacancy energy balance ===",
        f"  λ_v (grain count)   = {lam_count:.6e}",
        f"  λ_v (δF=0 balance)  = {lam_var:.6e}",
        f"  relative difference = {abs(lam_count-lam_var)/lam_count:.3e}",
        f"  γ₀^micro = 2λ_v     = {g0_micro:.6e}",
        "",
        "=== Two γ₀ prescriptions (different roles) ===",
        f"  γ₀^micro (vacancy):  {g0_micro:.4e}  → M_inner ≈ {r.m_inner_predicted_kg:.4e} kg",
        f"  γ₀^smooth (core reg): {smooth['gamma0_smooth']:.4f}  → M_inner ≈ {r.m_inner_smooth_kg:.4e} kg",
        f"    smooth targets ρ_center ≈ {smooth['rho_center_target']:.2f} (avoid GP collapse)",
        f"    NOT used for exterior Newton — both give N@ref ≈ {r.newton_at_ref_smooth:.4f}",
        "",
        "=== Preferred CH reading ===",
        "  Exterior gravity: α_G^hydro,ref (grain) + matched tail.",
        "  M_inner prediction: γ₀^micro from L_vac (microphysics).",
        "  GP solver stability: γ₀^smooth (phenomenological core regularization).",
        "",
        "=== Mass identification (Layer 2 postulate) ===",
        f"  M_grav (r_s ID)              = {r.m_grav_kg:.4e} kg",
        f"  M_inner (micro)              = {r.m_inner_predicted_kg:.4e} kg",
        f"  M_grav/M_inner               = {r.m_grav_over_m_inner_pred:.4e}",
        "  Link is NOT ∫(1-ρ)dV = M_grav; test both scales when ξ is lab-fixed.",
        "",
        "--- full microphysics report ---",
        format_report(r),
    ]

    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_defect_gamma0_variational.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_defect_gamma0_variational.txt'}")


if __name__ == "__main__":
    main()
