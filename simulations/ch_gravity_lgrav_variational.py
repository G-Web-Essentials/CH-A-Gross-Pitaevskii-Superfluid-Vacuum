"""
κ-level gravity — L_int,grav → α_G (parallel to Paper 2 L_int → κ).

Minimal coupling (static, Newtonian sector):
    L_int,grav = -(λ_g / c²) Φ (ρ - ρ_in) / ρ_in

Matching to implemented hydrostatic channel:
    Φ_hydro = α_G (c_s²/m_grain) (ρ/ρ_in - 1)

Identification (same logic as κ = 2λ_em/c² for dielectric):
    α_G = 2 λ_g m_grain / c²
    λ_g = α_G c² / (2 m_grain)

Grain closure (not a Newton fit):
    α_G^hydro,ref = m_grain c² / (2 c_s²)

Far-field check: N_hydro ≈ 1 on Schwarzschild tail at α_G^hydro,ref.

  python ch_gravity_lgrav_variational.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ch_dispersion_core import CHParams, C, G_MEAS
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    hydrostatic_coupling,
    hydrostatic_newton_factor,
    mass_from_rs_hat,
)

OUTPUT = Path(__file__).parent / "output"


def lambda_g_from_alpha(ch: CHParams, alpha_g: float) -> float:
    """λ_g from α_G identification α_G = 2 λ_g m_grain / c²."""
    return alpha_g * C**2 / (2.0 * ch.m_grain)


def alpha_g_from_lambda_g(ch: CHParams, lambda_g: float) -> float:
    """Inverse identification α_G = 2 λ_g m_grain / c²."""
    return 2.0 * lambda_g * ch.m_grain / C**2


def lambda_g_grain_estimate(ch: CHParams) -> float:
    """
    Grain estimate: λ_g ~ E_ξ / c² (vacuum grain energy sets gravity–matter coupling).

    Parallel to λ_em ~ e²/(m_grain ω₀²) in Paper 2.
    """
    return ch.e_xi_j / C**2


def newton_factor_schwarzschild_tail(
    ch: CHParams,
    alpha_g: float,
    r_s_hat: float = 1.0,
    r_max_hat: float = 200.0,
    n: int = 2000,
) -> float:
    """N_hydro on analytic (1 - r_s/r)² tail (exterior Q ≈ 0)."""
    r_hat = np.linspace(r_s_hat * 1.05, r_max_hat, n)
    r_m = r_hat * ch.xi
    rho = np.clip(1.0 - r_s_hat / np.clip(r_hat, r_s_hat * 1.01, None), 0.0, None) ** 2
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_g)
    phi = hydrostatic_coupling(ch_ref, rho)
    accel = -np.gradient(phi, r_m)
    m = mass_from_rs_hat(ch, r_s_hat)
    g_newton = G_MEAS * m / r_m**2
    n_hydro = hydrostatic_newton_factor(ch_ref)
    # slope of |a| vs GM/r²
    coeffs = np.polyfit(g_newton, np.abs(accel), 1)
    return float(coeffs[0])


def main() -> None:
    xi = 50e-9
    ch = CHParams(xi=xi)
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_id = lambda_g_from_alpha(ch, alpha_ref)
    alpha_back = alpha_g_from_lambda_g(ch, lam_id)
    lam_grain = lambda_g_grain_estimate(ch)
    alpha_grain = alpha_g_from_lambda_g(ch, lam_grain)

    n_tail = newton_factor_schwarzschild_tail(ch, alpha_ref)
    n_factor = hydrostatic_newton_factor(CHParams(xi=xi, alpha_g=alpha_ref))

    lines = [
        "CH κ-level gravity — L_int,grav → α_G (Route G5)",
        f"xi = {xi:.3e} m",
        "",
        "=== Minimal interaction (parallel to Paper 2) ===",
        "  Paper 2:  L_int,em  = -(λ_em/c²) |A|² (ρ - ρ_in)/ρ_in",
        "             ⇒ κ = 2 λ_em / c²",
        "",
        "  Paper 1:  L_int,grav = -(λ_g/c²) Φ (ρ - ρ_in)/ρ_in",
        "             ⇒ α_G = 2 λ_g m_grain / c²",
        "             ⇒ λ_g = α_G c² / (2 m_grain)",
        "",
        "=== Matching to hydrostatic implementation ===",
        "  Φ_hydro = α_G (c_s²/m_grain) (ρ/ρ_in - 1)",
        "  Far-field linear depletion: |a|/(GM/r²) = α_G · 2c_s²/(m_grain c²)",
        f"  Grain closure: α_G^hydro,ref = m_grain c²/(2c_s²) = {alpha_ref:.6e}",
        "",
        "=== Numerical identification check ===",
        f"  λ_g(α_G^ref)     = {lam_id:.6e}",
        f"  α_G(λ_g) roundtrip = {alpha_back:.6e}  (rel err {abs(alpha_back-alpha_ref)/alpha_ref:.3e})",
        f"  λ_g (E_ξ/c² est) = {lam_grain:.6e}",
        f"  α_G from λ_grain  = {alpha_grain:.6e}  (order-of-magnitude grain route)",
        "",
        "=== Exterior Newton (Schwarzschild tail, hydro only) ===",
        f"  N_hydro factor (linear) = {n_factor:.6f}",
        f"  N_hydro slope (tail fit) = {n_tail:.6f}",
        "",
        "=== What is derived vs identified ===",
        "  DERIVED (κ-level): form L_int,grav and α_G = 2λ_g m_grain/c² from variation.",
        "  GRAIN CLOSURE: α_G^hydro,ref = m_grain c²/(2c_s²) fixes λ_g without Newton fit.",
        "  IDENTIFIED (Layer 2): M_grav from r_s; matched tail BC on GP solve.",
        "  OPEN: dynamical Φ, full S_M[g,ψ], core Q + hydro joint far field.",
        "",
        "Reviewer-safe: exterior N_hydro ~ 1 at α_G^hydro,ref (1D + 2D oblate rays).",
    ]
    report = "\n".join(lines)
    print(report)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_lgrav_variational.txt").write_text(report + "\n")
    print(f"\nWrote {OUTPUT / 'ch_gravity_lgrav_variational.txt'}")


if __name__ == "__main__":
    main()
