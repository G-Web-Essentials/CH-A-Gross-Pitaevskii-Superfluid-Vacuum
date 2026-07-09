"""
Paper 2 — Phase 1b: perturbative parallel-plate TE mode sum for inhomogeneous ε(z).

Perfect conductors at z = 0, d; unperturbed TE modes ψ_n(z) = sin(nπz/d).
First-order fractional frequency shift (κ ≪ 1, static ε):

  δω_n / ω_n^(0) ≈ ∫₀¹ δ(u) sin²(nπu) du,
  δ(u) = ε(z)/ε₀ − 1  from coupling C1.

Casimir force response weighted by n² (leading UV-finite combination for
δF/F at fixed cutoff):

  (δF/F_Cas) ≈ Σ_{n=1}^{N} n² (δω_n/ω_n) / Σ_{n=1}^{N} n².

Also provides numeric ∂E/∂d from a truncated mode sum (cross-check).

See docs/paper-2/coupling-spec.md § Mode-sum method.
"""

from __future__ import annotations

import numpy as np

from ch_gpe_core import thomas_fermi_box_profile

HBAR = 1.054571817e-34
C = 299792458.0


def delta_epsilon_relative(
    u: np.ndarray,
    d_hat: float,
    chi_bulk: float,
    kappa: float,
) -> np.ndarray:
    """δε/ε₀ = κ χ_bulk (1 − ρ/ρ_in) on normalized coordinate u = z/d."""
    z_hat = u * d_hat
    rho = thomas_fermi_box_profile(z_hat, d_hat)
    return kappa * chi_bulk * (1.0 - rho)


def delta_epsilon_from_rho_norm(
    rho_norm: np.ndarray,
    chi_bulk: float,
    kappa: float,
) -> np.ndarray:
    """δε/ε₀ = κ χ_bulk (1 − ρ/ρ_in) from normalized density on u grid."""
    return kappa * chi_bulk * (1.0 - rho_norm)


def mode_frequency_shift_from_delta(
    n: int,
    delta: np.ndarray,
    u: np.ndarray,
) -> float:
    """δω_n / ω_n^(0) = ∫₀¹ δ(u) sin²(nπu) du."""
    integrand = delta * np.sin(n * np.pi * u) ** 2
    return float(np.trapz(integrand, u))


def mode_frequency_shift_fractional(
    n: int,
    d_hat: float,
    chi_bulk: float,
    kappa: float,
    n_u: int = 400,
) -> float:
    """
    δω_n / ω_n^(0) for TE mode n under small static δε(z).

    ∫₀^d δ(z) sin²(nπz/d) (2/d) dz = ∫₀¹ δ(u) sin²(nπu) du.
    """
    if chi_bulk <= 0.0 or d_hat < 0.04:
        return 0.0
    u = np.linspace(0.0, 1.0, n_u)
    delta = delta_epsilon_relative(u, d_hat, chi_bulk, kappa)
    return mode_frequency_shift_from_delta(n, delta, u)


def mode_frequency_shift_from_rho(
    n: int,
    rho_norm: np.ndarray,
    chi_bulk: float,
    kappa: float,
    u: np.ndarray | None = None,
) -> float:
    """δω_n / ω_n^(0) for arbitrary normalized ρ(u) on [0, 1]."""
    if chi_bulk <= 0.0:
        return 0.0
    if u is None:
        u = np.linspace(0.0, 1.0, len(rho_norm))
    delta = delta_epsilon_from_rho_norm(rho_norm, chi_bulk, kappa)
    return mode_frequency_shift_from_delta(n, delta, u)


def mode_sum_force_correction_from_rho(
    rho_norm: np.ndarray,
    chi_bulk: float,
    kappa: float,
    n_max: int = 800,
    weight_power: int = 2,
    u: np.ndarray | None = None,
) -> float:
    """UV-finite weighted mode sum for δF/F_Cas from ρ(u)/ρ_in profile."""
    if chi_bulk <= 0.0:
        return 0.0
    if u is None:
        u = np.linspace(0.0, 1.0, len(rho_norm))
    ns = np.arange(1, n_max + 1, dtype=float)
    shifts = np.array(
        [mode_frequency_shift_from_rho(int(n), rho_norm, chi_bulk, kappa, u) for n in ns]
    )
    weights = ns ** weight_power
    return float(np.sum(weights * shifts) / np.sum(weights))


def mode_sum_force_correction_te_tm_from_rho(
    rho_norm: np.ndarray,
    chi_bulk: float,
    kappa: float,
    n_max: int = 800,
    u: np.ndarray | None = None,
) -> float:
    """TE + TM leading-order from arbitrary ρ(u)."""
    return 2.0 * mode_sum_force_correction_from_rho(
        rho_norm, chi_bulk, kappa, n_max=n_max, weight_power=2, u=u
    )


def mode_sum_force_correction(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_max: int = 800,
    weight_power: int = 2,
) -> float:
    """
    UV-finite weighted mode sum for δF/F_Cas (perturbative, TE sector).

    weight_power = 2 matches leading Casimir force sensitivity (∂ω_n/∂d ∝ n).
    """
    if chi_bulk <= 0.0 or d_m <= 0.0:
        return 0.0
    d_hat = d_m / xi_m
    if d_hat < 0.04:
        return 0.0

    ns = np.arange(1, n_max + 1, dtype=float)
    shifts = np.array(
        [mode_frequency_shift_fractional(int(n), d_hat, chi_bulk, kappa) for n in ns]
    )
    weights = ns ** weight_power
    return float(np.sum(weights * shifts) / np.sum(weights))


def mode_sum_force_correction_te_tm(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_max: int = 800,
) -> float:
    """TE + TM leading-order: same sin² overlap for both polarizations → factor 2."""
    return 2.0 * mode_sum_force_correction(
        d_m, xi_m, chi_bulk, kappa, n_max=n_max, weight_power=2
    )


def casimir_energy_truncated(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_max: int = 500,
) -> float:
    """E = (ħ/2) Σ ω_n with ω_n = (nπc/d)(1 + shift_n). Truncated (not ζ-regularized)."""
    if d_m <= 0.0:
        return 0.0
    d_hat = d_m / xi_m
    energy = 0.0
    for n in range(1, n_max + 1):
        shift = mode_frequency_shift_fractional(n, d_hat, chi_bulk, kappa)
        omega_n = (n * np.pi * C / d_m) * (1.0 + 0.5 * shift)
        energy += 0.5 * HBAR * omega_n
    return energy


def force_from_energy_numeric(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_max: int = 500,
    h_rel: float = 1e-4,
) -> float:
    """F = −∂E/∂d via symmetric finite difference on truncated sum."""
    h = d_m * h_rel
    e_plus = casimir_energy_truncated(d_m + h, xi_m, chi_bulk, kappa, n_max)
    e_minus = casimir_energy_truncated(d_m - h, xi_m, chi_bulk, kappa, n_max)
    return -(e_plus - e_minus) / (2.0 * h)


def force_ratio_energy_numeric(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_max: int = 500,
) -> float:
    """F_pert / F_0 from truncated energy derivative (κ=0 reference)."""
    f_pert = force_from_energy_numeric(d_m, xi_m, chi_bulk, kappa, n_max)
    f_0 = force_from_energy_numeric(d_m, xi_m, chi_bulk, 0.0, n_max)
    if abs(f_0) < 1e-30:
        return 1.0
    return f_pert / f_0


def convergence_check(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_max_list: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Return [(n_max, correction)] for cutoff study."""
    if n_max_list is None:
        n_max_list = [50, 100, 200, 400, 800, 1200]
    return [
        (n, mode_sum_force_correction_te_tm(d_m, xi_m, chi_bulk, kappa, n_max=n))
        for n in n_max_list
    ]
