"""
Paper 2 — supersolid cavity energy functionals (Phase 2b/2c).

Profiles (dimensionless ẑ = z/ξ, d̂ = d/ξ, a_hat = a_vac/ξ):

  ρ_TF = tanh(ẑ) tanh(d̂ − ẑ)
  ρ    = ρ_TF [1 + η cos(2π ẑ / a_hat)]

Energy models:
  E_mod   — gradient + deviation from TF (baseline)
  E_phase — supersolid phase rigidity ∝ (2π/a_hat)²
  E_heal  — healing-length locking: prefer 2π/a_hat ≈ 1 (a_hat ≈ 2π)
"""

from __future__ import annotations

import numpy as np

from ch_gpe_core import thomas_fermi_box_profile

A_HYP_FACTOR = 2.0 * np.pi  # Paper 1: a_vac = 2πξ


def tf_bulk_integral(d_hat: float, n_pts: int = 400) -> float:
    """∫ ρ_TF dẑ — weight for phase / healing penalties."""
    z_hat = np.linspace(0.0, d_hat, n_pts)
    tf = thomas_fermi_box_profile(z_hat, d_hat)
    return float(np.trapz(tf, z_hat))


def supersolid_modulation_energy(
    d_hat: float,
    eta: float,
    a_vac_hat: float,
    n_pts: int = 400,
) -> float:
    """E_mod = ∫ [½|∂ρ/∂ẑ|² + ½(ρ − ρ_TF)²] dẑ."""
    z_hat = np.linspace(0.0, d_hat, n_pts)
    tf = thomas_fermi_box_profile(z_hat, d_hat)
    phase = 2.0 * np.pi * z_hat / a_vac_hat
    rho = tf * (1.0 + eta * np.cos(phase))
    rho = np.clip(rho, 0.0, None)
    grad = np.gradient(rho, z_hat)
    density = 0.5 * grad**2 + 0.5 * (rho - tf) ** 2
    return float(np.trapz(density, z_hat))


def phase_stiffness_energy(
    d_hat: float,
    a_vac_hat: float,
    k_phase: float,
    n_pts: int = 400,
) -> float:
    """
    E_phase = (K/2) ∫ |∂θ/∂ẑ|² ρ_TF dẑ,  θ = 2π ẑ/a_hat.

    Favors longer period (smaller phase gradient).
    """
    if k_phase <= 0.0 or a_vac_hat <= 0.0:
        return 0.0
    k_theta = 2.0 * np.pi / a_vac_hat
    bulk = tf_bulk_integral(d_hat, n_pts)
    return 0.5 * k_phase * k_theta**2 * bulk


def healing_lock_energy(
    d_hat: float,
    a_vac_hat: float,
    k_heal: float,
    q_target: float = 1.0,
    n_pts: int = 400,
) -> float:
    """
    E_heal = (λ/2) ∫ (2π/a_hat − q_target)² ρ_TF dẑ.

    q_target = 1 ⟺ a_hat = 2π (Paper 1 identification at ξ scale).
    """
    if k_heal <= 0.0 or a_vac_hat <= 0.0:
        return 0.0
    mismatch = (2.0 * np.pi / a_vac_hat) - q_target
    bulk = tf_bulk_integral(d_hat, n_pts)
    return 0.5 * k_heal * mismatch**2 * bulk


def total_supersolid_energy(
    d_hat: float,
    eta: float,
    a_vac_hat: float,
    k_phase: float = 0.0,
    k_heal: float = 0.0,
    q_target: float = 1.0,
    n_pts: int = 400,
) -> float:
    """E_mod + optional phase stiffness and healing-lock terms."""
    e = supersolid_modulation_energy(d_hat, eta, a_vac_hat, n_pts)
    e += phase_stiffness_energy(d_hat, a_vac_hat, k_phase, n_pts)
    e += healing_lock_energy(d_hat, a_vac_hat, k_heal, q_target, n_pts)
    return e
