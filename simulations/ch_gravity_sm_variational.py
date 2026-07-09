"""
Full S_M variational sketch — dynamical Φ coupled to GP + L_int,grav + L_vac.

Action (static / Newtonian sector):
  S = ∫ d³x [ L_GP[ψ] + L_Φ[Φ] + L_int,grav + L_vac ]

Euler–Lagrange:
  δS/δψ*  → GP + vacancy sink + Φ-coupling
  δS/δΦ   → Poisson-type equation sourcing from (ρ - ρ_in)/ρ_in

Quasi-static reduction (laptop v3):
  Slaved hydrostatic Φ_hydro + matched Schwarzschild BC; Q from Madelung.

Toy Poisson solve on Schwarzschild ρ validates Φ → Newtonian slope.

  python ch_gravity_sm_variational.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ch_dispersion_core import CHParams, C, G_MEAS
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    hydrostatic_coupling,
    mass_from_rs_hat,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha

OUTPUT = Path(__file__).parent / "output"


def poisson_phi_radial(
    r_m: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve ∇²Φ = S on uniform radial grid (spherical symmetry).

    Φ'' + (2/r)Φ' = S  →  d/dr(r² Φ') = r² S
    BC: Φ'(0) = 0 (regular), Φ(r_max) = 0
    """
    r = np.asarray(r_m, dtype=float)
    s = np.asarray(source, dtype=float)
    n = len(r)
    dr = r[1] - r[0]
    flux = np.zeros(n)
    for i in range(1, n):
        flux[i] = flux[i - 1] + 0.5 * (r[i] ** 2 * s[i] + r[i - 1] ** 2 * s[i - 1]) * dr
    dphi_dr = np.zeros(n)
    dphi_dr[1:] = flux[1:] / np.clip(r[1:] ** 2, 1e-30, None)
    dphi_dr[0] = dphi_dr[1]
    phi = np.zeros(n)
    for i in range(n - 2, -1, -1):
        phi[i] = phi[i + 1] - 0.5 * (dphi_dr[i] + dphi_dr[i + 1]) * dr
    phi[-1] = 0.0
    return phi, dphi_dr


def schwarzschild_rho(r_hat: np.ndarray, r_s_hat: float) -> np.ndarray:
    return np.clip(1.0 - r_s_hat / np.clip(r_hat, r_s_hat * 1.01, None), 0.0, None) ** 2


def newton_slope_from_phi(r_m: np.ndarray, dphi_dr: np.ndarray, mass_kg: float) -> float:
    """Fit |dΦ/dr| vs GM/r² on exterior annulus."""
    g_newton = G_MEAS * mass_kg / np.clip(r_m**2, 1e-60, None)
    mask = r_m > r_m[len(r_m) // 4]
    coeffs = np.polyfit(g_newton[mask], np.abs(dphi_dr[mask]), 1)
    return float(coeffs[0])


def derive_report(ch: CHParams) -> str:
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)

    return "\n".join(
        [
            "CH full S_M variational sketch (Route G5b — dynamical Φ)",
            f"xi = {ch.xi:.3e} m",
            "",
            "=== Action (static sector) ===",
            "  S = ∫ d³x [ L_GP + L_Φ + L_int,grav + L_vac ]",
            "",
            "  L_GP      = (ℏ²/2m)|∇ψ|² + (g/2)|ψ|⁴ − μ|ψ|²",
            "  L_Φ       = (Z_Φ/2c²) |∇Φ|²",
            "  L_int,grav= −(λ_g/c²) Φ (|ψ|² − ρ_in)/ρ_in",
            "  L_vac     = −λ_v ∫ γ(x)|ψ|² d³x   (→ sink −γ₀ e^{−r²/σ²} ψ)",
            "",
            "=== Euler–Lagrange ===",
            "  δS/δψ*:  μψ = [−ℏ²∇²/2m + g|ψ|²]ψ − γ(x)ψ",
            "            + (λ_g Φ / c² ρ_in) ψ",
            "",
            "  δS/δΦ:   −(Z_Φ/c²)∇²Φ = (λ_g/c²)(ρ − ρ_in)/ρ_in",
            "            ⇒ ∇²Φ = (λ_g/Z_Φ)(ρ − ρ_in)/ρ_in",
            "",
            "=== Identifications (κ-level, parallel Paper 2) ===",
            "  α_G = 2 λ_g m_grain / c²",
            f"  α_G^hydro,ref = {alpha_ref:.6e}",
            f"  λ_g = {lam_g:.6e}",
            "",
            "  Slaved hydrostatic limit (laptop v3):",
            "    Φ_hydro = α_G (c_s²/m_grain)(ρ/ρ_in − 1)  [algebraic, not Poisson solve]",
            "    Φ_total = Q[ψ] + Φ_hydro  on fixed ρ(r) from matched GP",
            "",
            "  Dynamical limit: solve Poisson for Φ with ρ from GP; far field → GM/r",
            "  when Z_Φ chosen so (λ_g/Z_Φ) matches Newtonian sourcing.",
            "",
            "=== What laptop implements today ===",
            "  GP + L_vac sink (γ₀) on inner ball; Dirichlet ρ_join on r_join;",
            "  Schwarzschild tail continuation; hydrostatic Φ on exterior;",
            "  grain α_G^hydro,ref (not Poisson inversion each step).",
            "",
            "=== Still open ===",
            "  Full δS/δg_μν; back-reaction of Φ on ψ beyond linear coupling;",
            "  joint core Q + dynamical Φ + rotation / 3D.",
        ]
    )


def main() -> None:
    xi = 50e-9
    rs_hat = 1.0
    ch = CHParams(xi=xi)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    mass_kg = mass_from_rs_hat(ch, rs_hat)

    # Toy Poisson: source ∝ (ρ/ρ_in − 1) with coupling tied to α_G
    # Choose Z_Φ so linear far-field matches hydrostatic slope:
    # For small deficit, Φ_hydro = α_G (c_s²/m)(ρ/ρ_in−1) solves slaved limit.
    # Poisson: ∇²Φ = A (ρ/ρ_in − 1); for ρ ≈ 1 − r_s/r, Φ ∝ GM/r when A fixed.
    r_hat = np.linspace(0.05, 300.0, 4000) * rs_hat
    r_m = r_hat * ch.xi
    rho = schwarzschild_rho(r_hat, rs_hat)
    deficit = rho - 1.0
    ch_ref = CHParams(xi=xi, alpha_g=alpha_ref)
    phi_hydro = hydrostatic_coupling(ch_ref, rho)
    dphi_hydro = np.gradient(phi_hydro, r_m)

    z_phi = 1.0
    source0 = (lam_g / (z_phi * ch.rho_in)) * deficit
    phi0, dphi0 = poisson_phi_radial(r_m, source0)
    mask = r_m > 5.0 * ch.xi * rs_hat
    scale = float(
        np.median(np.abs(dphi_hydro[mask]))
        / max(np.median(np.abs(dphi0[mask])), 1e-99)
    )
    source = source0 * scale
    phi, dphi_dr = poisson_phi_radial(r_m, source)

    n_poisson = newton_slope_from_phi(r_m, dphi_dr, mass_kg)
    n_hydro = newton_slope_from_phi(r_m, dphi_hydro, mass_kg)

    report = derive_report(ch)
    report += "\n\n=== Toy Poisson on Schwarzschild ρ ===\n"
    report += f"  N_slope (dynamical Φ from Poisson) = {n_poisson:.4f}\n"
    report += f"  N_slope (slaved Φ_hydro, laptop)    = {n_hydro:.4f}\n"
    report += (
        "  Poisson toy uses same ρ(r); full S_M requires self-consistent ψ+Φ solve.\n"
    )

    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_sm_variational.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_gravity_sm_variational.txt'}")


if __name__ == "__main__":
    main()
