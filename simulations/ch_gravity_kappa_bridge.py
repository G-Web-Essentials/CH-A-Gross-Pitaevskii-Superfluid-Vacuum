"""
Gravity ↔ Paper 2 κ bridge — parallel variational routes (G1, G2, κ).

Paper 2:  L_int = -(λ_em/c²)|A|²(ρ-ρ_in)/ρ_in  →  κ = 2λ_em/c²
Paper 1 G1: α_G^hydro,ref = m_grain c² / (2 c_s²)  (hydrostatic δρ coupling)
Paper 1 G2: L_vac = -λ_v ∫ γ|ψ|²  →  γ₀ = 2λ_v

κ-level S_M (future): couple metric/curvature to δρ like EM couples to (ρ-ρ_in)/ρ_in.

  python ch_gravity_kappa_bridge.py
"""

from __future__ import annotations

from pathlib import Path

from ch_defect_microphysics import gamma0_from_microphysics, lambda_v_vacancy_coupling
from ch_dispersion_core import CHParams, C
from ch_gpe_gravity import alpha_g_required_for_hydrostatic_newton

OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    xi = 50e-9
    sigma = 0.6
    ch = CHParams(xi=xi)

    alpha_g = alpha_g_required_for_hydrostatic_newton(ch)
    lam_v = lambda_v_vacancy_coupling(ch, sigma)
    g0 = gamma0_from_microphysics(ch, sigma)

    lines = [
        "CH gravity ↔ Paper 2 κ bridge (variational parallel routes)",
        f"xi = {xi:.3e} m, sigma = {sigma}",
        "",
        "Paper 2 EM (Casimir / dielectric):",
        "  L_int = -(lambda_em/c^2) |A|^2 (rho - rho_in)/rho_in",
        "  Variation → dielectric shift ∝ (1 - rho/rho_in)",
        "  Identification: kappa = 2 lambda_em / c^2",
        "",
        "Paper 1 gravity G1 (hydrostatic, DONE on laptop):",
        "  Phi_hydro = alpha_G (c_s^2/m_grain) (rho/rho_in - 1)",
        "  Far-field linear depletion → |a|/(GM/r^2) = alpha_G * 2 c_s^2 / (m_grain c^2)",
        "  Grain closure: alpha_G^hydro,ref = m_grain c^2 / (2 c_s^2)",
        f"  => alpha_G^hydro,ref = {alpha_g:.6e}",
        "",
        "Paper 1 gravity G2 (vacancy sink, ansatz + EL):",
        "  L_vac = -lambda_v ∫ gamma(x) |psi|^2 d^3x",
        "  Variation → sink term -2 lambda_v gamma psi  →  gamma_0 = 2 lambda_v",
        f"  => lambda_v = {lam_v:.4e}, gamma_0^micro = {g0:.4e}",
        "",
        "Structural parallel:",
        "  kappa  : 2 lambda_em / c^2     (EM ↔ vacuum polarization)",
        "  gamma_0: 2 lambda_v            (vacancy ↔ condensate removal)",
        "  alpha_G: m_grain c^2/(2 c_s^2) (hydrostatic ↔ Newton strength)",
        "",
        "kappa-level S_M (OPEN — research track):",
        "  Target: L_grav or L_int,grav = -(lambda_g/c^2) Phi_g (rho-rho_in)/rho_in",
        "          or curvature scalar coupled to chi_grad * (1 - rho/rho_in),",
        "          with lambda_g fixed by grain counting like alpha_G^hydro,ref.",
        "  Full 3D action S_M[g, psi] not yet derived; laptop uses matched BC +",
        "  hydrostatic channel on fixed rho(r) from GP (Layer 2 identification).",
        "",
        "Reviewer-safe claim today:",
        "  alpha_G^hydro,ref is grain-derived; exterior N_hydro ~ 1 (1D + 2D rays).",
        "  gamma_0^micro predicts M_inner; kappa-level gravity is future work.",
    ]
    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_kappa_bridge.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_gravity_kappa_bridge.txt'}")


if __name__ == "__main__":
    main()
