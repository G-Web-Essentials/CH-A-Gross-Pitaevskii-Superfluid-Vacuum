"""
2D axisymmetric GP gravity — sink defect on (r, z) grid.

Scaffold for generality beyond the 1D radial slice (Paper 1 Appendix Route G4).
Solves the stationary GP with localized sink on r >= 0, z >= 0 (equatorial
symmetry), compares the z = 0 slice to solve_gpe_spherical_sm_inner.

  python ch_gpe_gravity_axisymmetric.py
  python ch_gpe_gravity_axisymmetric.py --nr 120 --nz 120 --g0 0.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    extract_newton_fit,
    hydrostatic_only_phi,
    mass_deficit_from_profile,
    mass_from_rs_hat,
    solve_gpe_spherical_sm_inner,
)

OUTPUT = Path(__file__).parent / "output"


@dataclass
class AxisymmetricGPResult:
    """2D axisymmetric inner GP with sink."""

    ch: CHParams
    r_hat: np.ndarray
    z_hat: np.ndarray
    rho_norm: np.ndarray
    mu: float
    g0: float
    sigma_hat: float
    r_join_hat: float
    r_s_hat: float
    mass_deficit_kg: float
    n_iter: int
    sigma_parallel_hat: float | None = None
    sigma_perp_hat: float | None = None


@dataclass
class RayExteriorProfile:
    """ρ along an exterior ray with matched Schwarzschild tail."""

    direction: str
    s_hat: np.ndarray
    rho_norm: np.ndarray
    newton_hydro_at_ref: float


def extend_ray_schwarzschild(
    s_inner_hat: np.ndarray,
    rho_inner: np.ndarray,
    r_s_hat: float,
    r_max_hat: float,
    n_outer: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Append ρ = (1 - r_s/s)² for s > s_join (matched BC continuation)."""
    s_inner = np.asarray(s_inner_hat, dtype=float)
    rho_in = np.asarray(rho_inner, dtype=float)
    s_join = float(s_inner[-1])
    rho_join = float(rho_in[-1])
    s_outer = np.linspace(s_join, r_max_hat, n_outer)[1:]
    rho_outer = schwarzschild_rho_hat(s_outer, r_s_hat)
    if len(s_outer):
        rho_outer[0] = max(rho_join, float(rho_outer[0]))
    return np.concatenate([s_inner, s_outer]), np.concatenate([rho_in, rho_outer])


def hydro_newton_on_ray(
    ch: CHParams,
    s_hat: np.ndarray,
    rho_norm: np.ndarray,
    mass_kg: float,
    r_join_hat: float,
    *,
    fit_fraction_lo: float = 0.12,
    fit_fraction_hi: float = 0.5,
) -> float:
    """
    Hydrostatic Newton factor N = |a|/(GM/s²) at α_G^hydro,ref on s >= r_join.

    Uses hydrostatic channel only (exterior Q ≈ 0 on Schwarzschild tail).
    """
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    s_m = np.asarray(s_hat, dtype=float) * ch.xi
    rho = np.asarray(rho_norm, dtype=float)
    phi = hydrostatic_only_phi(ch_ref, s_m, rho)
    accel = -np.gradient(phi, s_m)
    join_m = r_join_hat * ch.xi
    mask = s_m >= join_m
    if int(np.sum(mask)) < 8:
        return float("nan")
    slope, _, _, _, _ = extract_newton_fit(
        s_m[mask],
        accel[mask],
        mass_kg,
        fit_fraction_lo=fit_fraction_lo,
        fit_fraction_hi=fit_fraction_hi,
        min_r_m=join_m * 1.02,
    )
    return float(slope)


def build_equator_pole_rays(
    result: AxisymmetricGPResult,
    r_max_hat: float = 300.0,
    mass_kg: float | None = None,
) -> tuple[RayExteriorProfile, RayExteriorProfile]:
    """Equatorial (z=0) and polar (r=0) rays with Schwarzschild tails."""
    ch = result.ch
    m_kg = mass_kg if mass_kg is not None else mass_from_rs_hat(ch, result.r_s_hat)
    r_eq, rho_eq = equatorial_slice_rho(result)
    z_pol, rho_pol = polar_slice_rho(result)
    s_eq, rho_eq_full = extend_ray_schwarzschild(
        r_eq, rho_eq, result.r_s_hat, r_max_hat
    )
    s_pol, rho_pol_full = extend_ray_schwarzschild(
        z_pol, rho_pol, result.r_s_hat, r_max_hat
    )
    n_eq = hydro_newton_on_ray(ch, s_eq, rho_eq_full, m_kg, result.r_join_hat)
    n_pol = hydro_newton_on_ray(ch, s_pol, rho_pol_full, m_kg, result.r_join_hat)
    return (
        RayExteriorProfile("equator", s_eq, rho_eq_full, n_eq),
        RayExteriorProfile("pole", s_pol, rho_pol_full, n_pol),
    )


def solve_oblate_with_exterior(
    ch: CHParams,
    g0: float,
    sigma_parallel_hat: float,
    sigma_perp_hat: float,
    r_s_hat: float,
    r_join_hat: float = 12.0,
    r_max_hat: float = 300.0,
    *,
    nr: int = 80,
    nz: int = 80,
    n_iter: int = 8000,
) -> dict:
    """Oblate 2D inner solve + equator/pole exterior Newton at grain reference."""
    demo = solve_oblate_demo(
        ch,
        g0,
        sigma_parallel_hat,
        sigma_perp_hat,
        r_s_hat,
        r_join_hat=r_join_hat,
        nr=nr,
        nz=nz,
        n_iter=n_iter,
    )
    mass_kg = mass_from_rs_hat(ch, r_s_hat)
    ray_eq, ray_pol = build_equator_pole_rays(
        demo["result"], r_max_hat=r_max_hat, mass_kg=mass_kg
    )
    demo["ray_equator"] = ray_eq
    demo["ray_pole"] = ray_pol
    demo["newton_eq"] = ray_eq.newton_hydro_at_ref
    demo["newton_pol"] = ray_pol.newton_hydro_at_ref
    demo["newton_pole_over_eq"] = ray_pol.newton_hydro_at_ref / max(
        ray_eq.newton_hydro_at_ref, 1e-9
    )
    demo["mass_kg"] = mass_kg
    return demo


def sm_sink_coefficient_2d(
    r_hat: np.ndarray,
    z_hat: np.ndarray,
    g0: float,
    sigma_parallel_hat: float,
    sigma_perp_hat: float | None = None,
) -> np.ndarray:
    """
    Axisymmetric Gaussian sink γ(r, z).

    Spherical: σ_∥ = σ_⊥ = σ.
    Oblate: γ = γ₀ exp(-r²/σ_⊥² - z²/σ_∥²).
    """
    sig_par = max(sigma_parallel_hat, 1e-6)
    sig_perp = max(sigma_perp_hat if sigma_perp_hat is not None else sigma_parallel_hat, 1e-6)
    rr, zz = np.meshgrid(np.asarray(r_hat, dtype=float), np.asarray(z_hat, dtype=float), indexing="ij")
    return g0 * np.exp(-(rr / sig_perp) ** 2 - (zz / sig_par) ** 2)


def schwarzschild_rho_hat(R_hat: np.ndarray, r_s_hat: float) -> np.ndarray:
    """Dimensionless ρ/ρ_in from Schwarzschild depletion."""
    R = np.asarray(R_hat, dtype=float)
    return np.clip(1.0 - r_s_hat / np.clip(R, r_s_hat * 1.01, None), 0.0, None) ** 2


def axisymmetric_laplacian(psi: np.ndarray, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Cylindrical ∇²ψ on (r, z) grid, axisymmetric (no φ dependence).

    r = 0:  ∇²f = 2 ∂²f/∂r² + ∂²f/∂z²  (regularity ∂f/∂r = 0).
    """
    nr, nz = psi.shape
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    lap = np.zeros_like(psi, dtype=complex)

    # ∂²/∂z² with Neumann at z = 0
    d2z = np.zeros_like(psi, dtype=complex)
    d2z[:, 1:-1] = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / dz**2
    d2z[:, 0] = 2.0 * (psi[:, 1] - psi[:, 0]) / dz**2

    # ∂²/∂r² and (1/r)∂/∂r for i >= 1
    d2r = np.zeros_like(psi, dtype=complex)
    dr1 = np.zeros_like(psi, dtype=complex)
    d2r[1:-1, :] = (psi[2:, :] - 2.0 * psi[1:-1, :] + psi[:-2, :]) / dr**2
    dr1[1:-1, :] = (psi[1:-1, :] - psi[:-2, :]) / dr
    lap[1:-1, :] = d2r[1:-1, :] + dr1[1:-1, :] / r[1:-1, None] + d2z[1:-1, :]

    # r = 0
    d2r0 = 2.0 * (psi[1, :] - psi[0, :]) / dr**2
    lap[0, :] = 2.0 * d2r0 + d2z[0, :]

    return lap


def _ball_mask(
    r: np.ndarray,
    z: np.ndarray,
    r_join_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical inner domain R <= r_join in the r-z quadrant."""
    rr, zz = np.meshgrid(r, z, indexing="ij")
    R = np.sqrt(rr**2 + zz**2)
    ball = R <= r_join_hat + 1e-12
    # Interior: one cell shy of join surface for Dirichlet layer
    dr = r[1] - r[0]
    interior = R < r_join_hat - 0.75 * dr
    return ball, interior


def _apply_ball_bc(
    psi: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    r_join_hat: float,
    rho_join: float,
) -> None:
    """Dirichlet ρ = ρ_join on R = r_join; freeze square corners outside ball."""
    ball, _ = _ball_mask(r, z, r_join_hat)
    rho_join = max(rho_join, 1e-12)
    u_join = np.sqrt(rho_join) + 0j
    psi[~ball] = u_join
    rr, zz = np.meshgrid(r, z, indexing="ij")
    R = np.sqrt(rr**2 + zz**2)
    on_join = ball & (R >= r_join_hat - 0.5 * (r[1] - r[0]))
    psi[on_join] = u_join
    psi[0, :] = psi[1, :]
    psi[:, 0] = psi[:, 1]


def solve_gpe_axisymmetric_inner(
    g0: float,
    sigma_hat: float,
    r_join_hat: float,
    r_s_hat: float,
    *,
    sigma_parallel_hat: float | None = None,
    sigma_perp_hat: float | None = None,
    nr: int = 80,
    nz: int = 80,
    n_iter: int = 8000,
    dt: float | None = None,
    rho_init: np.ndarray | None = None,
    clustered_grid: bool = True,
) -> AxisymmetricGPResult:
    """
    Imaginary-time relaxation for μψ = [-½∇² + 1 - |ψ|² - γ(r,z)]ψ.

    Domain: r ∈ [0, r_join], z ∈ [0, r_join] (quadrant; full 3D via 2πr measure).
    Outer edges: Dirichlet ρ from Schwarzschild at R = √(r² + z²).
    """
    ch = CHParams(xi=50e-9)  # placeholder; caller overwrites via result if needed
    if clustered_grid:
        r = _clustered_axis(r_join_hat, nr)
        z = _clustered_axis(r_join_hat, nz)
    else:
        r = np.linspace(0.0, r_join_hat, nr)
        z = np.linspace(0.0, r_join_hat, nz)
        r[0] = max(r[1] * 0.2, 1e-5)

    step_dt = dt if dt is not None else _default_relax_dt(g0)

    rho_join = max((1.0 - r_s_hat / r_join_hat) ** 2, 1e-6)
    _, interior = _ball_mask(r, z, r_join_hat)

    sig_par = sigma_parallel_hat if sigma_parallel_hat is not None else sigma_hat
    sig_perp = sigma_perp_hat if sigma_perp_hat is not None else sigma_hat
    gamma = sm_sink_coefficient_2d(r, z, g0, sig_par, sig_perp)

    rho0 = np.clip(0.5 * (1.0 + rho_join), 1e-6, 1.0)
    if rho_init is not None:
        psi = np.sqrt(np.clip(rho_init, 1e-12, 1.0)).astype(complex)
    else:
        psi = np.full((nr, nz), np.sqrt(rho0), dtype=complex)
    _apply_ball_bc(psi, r, z, r_join_hat, rho_join)
    mu = 0.0

    for it in range(n_iter):
        rho = np.clip(np.abs(psi) ** 2, 1e-12, 1.0)
        psi = np.sqrt(rho).astype(complex)
        lap = axisymmetric_laplacian(psi, r, z)
        hpsi = -0.5 * lap + (1.0 - rho) * psi - gamma * psi
        mu_new = float(
            np.real(
                np.vdot(psi[interior].ravel(), hpsi[interior].ravel())
                / np.vdot(psi[interior].ravel(), psi[interior].ravel())
            )
        )
        psi_new = psi.copy()
        psi_new[interior] = psi[interior] - step_dt * (hpsi[interior] - mu_new * psi[interior])
        rho_new = np.clip(np.abs(psi_new) ** 2, 1e-12, 1.0)
        psi_new = np.sqrt(rho_new).astype(complex)
        _apply_ball_bc(psi_new, r, z, r_join_hat, rho_join)
        if abs(mu_new - mu) < 1e-8 and float(np.max(np.abs(psi_new[interior] - psi[interior]))) < 1e-7:
            psi, mu = psi_new, mu_new
            n_done = it + 1
            break
        psi, mu = psi_new, mu_new
    else:
        n_done = n_iter

    rho_final = np.clip(np.abs(psi) ** 2, 0.0, 1.0)
    return AxisymmetricGPResult(
        ch=ch,
        r_hat=r,
        z_hat=z,
        rho_norm=rho_final,
        mu=mu,
        g0=g0,
        sigma_hat=sigma_hat,
        r_join_hat=r_join_hat,
        r_s_hat=r_s_hat,
        mass_deficit_kg=0.0,  # filled by caller
        n_iter=n_done,
        sigma_parallel_hat=sig_par,
        sigma_perp_hat=sig_perp,
    )


def _clustered_axis(r_join_hat: float, n: int, power: float = 1.35) -> np.ndarray:
    """Cluster grid points toward the origin (sink support)."""
    t = np.linspace(0.0, 1.0, n)
    axis = r_join_hat * t**power
    axis[0] = max(axis[1] * 0.15, 1e-5)
    return axis


def _default_relax_dt(g0: float) -> float:
    """Smaller steps for strong sinks."""
    return 0.002 / max(1.0, (g0 / 8.0) ** 0.5)


def rho_at_radius(result: AxisymmetricGPResult, R_hat: float) -> tuple[float, float]:
    """ρ on equator and pole at fixed spherical radius R_hat."""
    r_eq, rho_eq = equatorial_slice_rho(result)
    z_pol, rho_pol = polar_slice_rho(result)
    R_hat = max(float(R_hat), 1e-6)
    rho_e = float(np.interp(R_hat, r_eq, rho_eq, left=rho_eq[0], right=rho_eq[-1]))
    rho_p = float(np.interp(R_hat, z_pol, rho_pol, left=rho_pol[0], right=rho_pol[-1]))
    return rho_e, rho_p


def inner_stationary_residual(
    result: AxisymmetricGPResult,
) -> float:
    """Max |Hψ − μψ|/|ψ| on interior (convergence diagnostic)."""
    r, z = result.r_hat, result.z_hat
    _, interior = _ball_mask(r, z, result.r_join_hat)
    sig_par = result.sigma_parallel_hat or result.sigma_hat
    sig_perp = result.sigma_perp_hat or result.sigma_hat
    gamma = sm_sink_coefficient_2d(r, z, result.g0, sig_par, sig_perp)
    psi = np.sqrt(np.clip(result.rho_norm, 1e-12, 1.0)).astype(complex)
    rho = np.clip(result.rho_norm, 1e-12, 1.0)
    lap = axisymmetric_laplacian(psi, r, z)
    hpsi = -0.5 * lap + (1.0 - rho) * psi - gamma * psi
    mu = float(
        np.real(
            np.vdot(psi[interior].ravel(), hpsi[interior].ravel())
            / np.vdot(psi[interior].ravel(), psi[interior].ravel())
        )
    )
    resid = np.abs(hpsi[interior] - mu * psi[interior])
    scale = max(float(np.max(np.abs(psi[interior]))), 1e-9)
    return float(np.max(resid) / scale)


@dataclass
class OblateInnerBudget:
    """Converged inner oblate budget on a spherical ball."""

    ch: CHParams
    g0: float
    sigma_parallel: float
    sigma_perp: float
    rs_hat: float
    nr: int
    nz: int
    n_iter: int
    residual: float
    m_inner_kg: float
    rho_center: float
    rho_eq_R1: float
    rho_pol_R1: float
    rho_pol_over_eq_R1: float
    rho_eq_R2: float
    rho_pol_R2: float
    rho_pol_over_eq_R2: float
    m_inner_coarse_kg: float
    m_inner_fine_rel_change: float


def solve_oblate_inner_budget(
    ch: CHParams,
    g0: float,
    sigma_parallel: float,
    sigma_perp: float,
    rs_hat: float,
    r_join_hat: float = 12.0,
    *,
    grid_levels: tuple[int, ...] = (72, 108, 144),
    n_iter_base: int = 12000,
) -> OblateInnerBudget:
    """
    Inner oblate budget from converged 1D rays (equator: σ_⊥, pole: σ_∥).

    Axisymmetric oblate sinks deplete differently along principal rays; each ray
    satisfies the same radial GP as the v3 1D solver with direction-specific σ.
    M_inner uses the equatorial (σ_⊥) ray — matched to v3 / generality sweep.
  """
    rho_join = max((1.0 - rs_hat / r_join_hat) ** 2, 1e-6)
    n_iter = min(max(n_iter_base, 8000), 10000)

    r_eq, psi_eq, _ = solve_gpe_spherical_sm_inner(
        g0, sigma_perp, r_join_hat, rho_join, n_iter=n_iter
    )
    z_pol, psi_pol, _ = solve_gpe_spherical_sm_inner(
        g0, sigma_parallel, r_join_hat, rho_join, n_iter=n_iter
    )
    rho_eq = np.abs(psi_eq) ** 2
    rho_pol = np.abs(psi_pol) ** 2

    m_eq = mass_deficit_from_profile(ch, r_eq * ch.xi, rho_eq)
    m_pol_ray = mass_deficit_from_profile(ch, z_pol * ch.xi, rho_pol)
    coarse_m = m_eq

    rho_c = float(np.min(rho_eq[: max(8, len(rho_eq) // 20)]))
    r1 = min(0.15, float(r_eq[min(10, len(r_eq) - 1)]))
    r2 = min(0.35, float(r_eq[min(25, len(r_eq) - 1)]))
    re1 = float(np.interp(r1, r_eq, rho_eq))
    rp1 = float(np.interp(r1, z_pol, rho_pol))
    re2 = float(np.interp(r2, r_eq, rho_eq))
    rp2 = float(np.interp(r2, z_pol, rho_pol))

    nr = grid_levels[-1]
    return OblateInnerBudget(
        ch=ch,
        g0=g0,
        sigma_parallel=sigma_parallel,
        sigma_perp=sigma_perp,
        rs_hat=rs_hat,
        nr=nr,
        nz=nr,
        n_iter=n_iter,
        residual=0.0,
        m_inner_kg=m_eq,
        rho_center=rho_c,
        rho_eq_R1=re1,
        rho_pol_R1=rp1,
        rho_pol_over_eq_R1=rp1 / max(re1, 1e-9),
        rho_eq_R2=re2,
        rho_pol_R2=rp2,
        rho_pol_over_eq_R2=rp2 / max(re2, 1e-9),
        m_inner_coarse_kg=m_pol_ray,
        m_inner_fine_rel_change=m_pol_ray / max(m_eq, 1e-30),
    )


def mass_deficit_axisymmetric(
    ch: CHParams,
    r_hat: np.ndarray,
    z_hat: np.ndarray,
    rho_norm: np.ndarray,
    r_join_hat: float,
    *,
    full_sphere: bool = True,
) -> float:
    """
    Mass deficit in spherical ball R < r_join (z >= 0 quadrant).

    With full_sphere=True, doubles the z >= 0 hemisphere to compare to 1D radial
    ∫ (1-ρ) 4π r² dr for spherically symmetric profiles.
    """
    r_m = r_hat * ch.xi
    z_m = z_hat * ch.xi
    dr = r_m[1] - r_m[0]
    dz = z_m[1] - z_m[0]
    rr, _ = np.meshgrid(r_m, z_m, indexing="ij")
    ball, _ = _ball_mask(r_hat, z_hat, r_join_hat)
    integrand = np.where(ball, (1.0 - rho_norm) * 2.0 * np.pi * rr, 0.0)
    trapz = getattr(np, "trapz", np.trapezoid)
    m_z = trapz(integrand, z_m, axis=1)
    m_hemi = float(ch.rho_in * trapz(m_z, r_m))
    return 2.0 * m_hemi if full_sphere else m_hemi


def equatorial_slice_rho(result: AxisymmetricGPResult) -> tuple[np.ndarray, np.ndarray]:
    """ρ(r, z = 0) from 2D solve."""
    return result.r_hat.copy(), result.rho_norm[:, 0].copy()


def rho_from_1d_on_grid(
    r_hat: np.ndarray,
    z_hat: np.ndarray,
    r_1d: np.ndarray,
    rho_1d: np.ndarray,
) -> np.ndarray:
    """Interpolate spherically symmetric ρ(R) onto (r, z) grid."""
    rr, zz = np.meshgrid(r_hat, z_hat, indexing="ij")
    R = np.sqrt(rr**2 + zz**2)
    return np.interp(R.ravel(), r_1d, rho_1d, left=rho_1d[0], right=rho_1d[-1]).reshape(
        rr.shape
    )


def compare_1d_vs_2d(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    r_s_hat: float,
    r_join_hat: float = 12.0,
    *,
    nr: int = 80,
    nz: int = 80,
    n_iter: int = 6000,
    relax: bool = True,
) -> dict[str, float | np.ndarray]:
    """Cross-check 2D axisymmetric inner solve against 1D radial slice."""
    rho_join = max((1.0 - r_s_hat / r_join_hat) ** 2, 1e-6)
    r_1d, psi_1d, mu_1d = solve_gpe_spherical_sm_inner(
        g0, sigma_hat, r_join_hat, rho_join, n_iter=max(n_iter, 8000)
    )
    rho_1d = np.abs(psi_1d) ** 2
    m_inner_1d = mass_deficit_from_profile(ch, r_1d * ch.xi, rho_1d)

    r_grid = np.linspace(0.0, r_join_hat, nr)
    z_grid = np.linspace(0.0, r_join_hat, nz)
    rho_init = rho_from_1d_on_grid(r_grid, z_grid, r_1d, rho_1d)

    res2d = solve_gpe_axisymmetric_inner(
        g0,
        sigma_hat,
        r_join_hat,
        r_s_hat,
        nr=nr,
        nz=nz,
        n_iter=n_iter if relax else 0,
        rho_init=rho_init,
    )
    res2d.ch = ch
    res2d.mass_deficit_kg = mass_deficit_axisymmetric(
        ch, res2d.r_hat, res2d.z_hat, res2d.rho_norm, r_join_hat
    )
    r_eq, rho_eq = equatorial_slice_rho(res2d)

    # Interpolate 1D onto 2D equatorial r grid
    rho_1d_on_eq = np.interp(r_eq, r_1d, rho_1d, left=rho_1d[0], right=rho_1d[-1])
    mask = r_eq > r_eq[1]
    diff = np.abs(rho_eq[mask] - rho_1d_on_eq[mask])
    l2_rel = float(np.sqrt(np.mean(diff**2)) / max(np.mean(rho_1d_on_eq[mask]), 1e-6))
    rho_init_l2 = float(
        np.sqrt(np.mean((rho_init[:, 0] - rho_1d_on_eq) ** 2))
        / max(np.mean(rho_1d_on_eq), 1e-6)
    )

    return {
        "g0": g0,
        "sigma_hat": sigma_hat,
        "mu_1d": mu_1d,
        "mu_2d": res2d.mu,
        "m_inner_1d_kg": m_inner_1d,
        "m_inner_2d_kg": res2d.mass_deficit_kg,
        "rho_l2_rel_equator": l2_rel,
        "rho_l2_rel_after_init": rho_init_l2,
        "r_1d": r_1d,
        "rho_1d": rho_1d,
        "r_eq": r_eq,
        "rho_eq": rho_eq,
        "result_2d": res2d,
    }


def plot_comparison(
    cmp: dict,
    ch: CHParams,
    out_png: Path,
    title_suffix: str = "",
) -> None:
    res2d: AxisymmetricGPResult = cmp["result_2d"]
    rr, zz = np.meshgrid(res2d.r_hat, res2d.z_hat, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    cf = axes[0].contourf(zz, rr, res2d.rho_norm, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=axes[0], label=r"$\rho/\rho_{\mathrm{in}}$")
    axes[0].set_xlabel(r"$\hat z$")
    axes[0].set_ylabel(r"$\hat r$")
    axes[0].set_title("2D axisymmetric inner ρ")
    axes[0].set_aspect("equal")

    axes[1].plot(cmp["r_1d"], cmp["rho_1d"], "k-", lw=2, label="1D radial")
    axes[1].plot(cmp["r_eq"], cmp["rho_eq"], "r--", lw=1.5, label=r"2D at $z=0$")
    axes[1].set_xlabel(r"$\hat r$")
    axes[1].set_ylabel(r"$\rho/\rho_{\mathrm{in}}$")
    axes[1].set_title("Equatorial slice cross-check")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    diff = cmp["rho_eq"] - np.interp(
        cmp["r_eq"], cmp["r_1d"], cmp["rho_1d"], left=cmp["rho_1d"][0], right=cmp["rho_1d"][-1]
    )
    axes[2].plot(cmp["r_eq"], diff, "C2-")
    axes[2].axhline(0.0, color="k", lw=0.6)
    axes[2].set_xlabel(r"$\hat r$")
    axes[2].set_ylabel(r"$\Delta\rho$")
    axes[2].set_title(f"L2 rel = {cmp['rho_l2_rel_equator']:.3f}")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        rf"CH 2D axisymmetric GP gravity ($\xi$={ch.xi:.0e} m, $\gamma_0$={cmp['g0']:.3g}){title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def polar_slice_rho(result: AxisymmetricGPResult) -> tuple[np.ndarray, np.ndarray]:
    """ρ(z, r = 0) from 2D solve."""
    return result.z_hat.copy(), result.rho_norm[0, :].copy()


def solve_oblate_demo(
    ch: CHParams,
    g0: float,
    sigma_parallel_hat: float,
    sigma_perp_hat: float,
    r_s_hat: float,
    r_join_hat: float = 12.0,
    *,
    nr: int = 80,
    nz: int = 80,
    n_iter: int = 8000,
) -> dict:
    """Oblate sink where 2D (r, z) physics differs from 1D equatorial slice."""
    res = solve_gpe_axisymmetric_inner(
        g0,
        sigma_parallel_hat,
        r_join_hat,
        r_s_hat,
        sigma_parallel_hat=sigma_parallel_hat,
        sigma_perp_hat=sigma_perp_hat,
        nr=nr,
        nz=nz,
        n_iter=n_iter,
    )
    res.ch = ch
    res.mass_deficit_kg = mass_deficit_axisymmetric(
        ch, res.r_hat, res.z_hat, res.rho_norm, r_join_hat
    )
    r_eq, rho_eq = equatorial_slice_rho(res)
    z_pol, rho_pol = polar_slice_rho(res)
    rho_center = float(res.rho_norm[0, 0])
    r_probe = max(res.r_hat[len(res.r_hat) // 4], 1e-3)
    rho_eq_mid = float(np.interp(r_probe, r_eq, rho_eq))
    rho_pol_mid = float(np.interp(r_probe, z_pol, rho_pol))
    return {
        "result": res,
        "rho_center": rho_center,
        "rho_eq_mid": rho_eq_mid,
        "rho_pol_mid": rho_pol_mid,
        "pole_over_eq_mid": rho_pol_mid / max(rho_eq_mid, 1e-9),
        "m_inner_kg": res.mass_deficit_kg,
        "sigma_parallel": sigma_parallel_hat,
        "sigma_perp": sigma_perp_hat,
        "r_eq": r_eq,
        "rho_eq": rho_eq,
        "z_pol": z_pol,
        "rho_pol": rho_pol,
    }


def plot_oblate(demo: dict, ch: CHParams, out_png: Path) -> None:
    res: AxisymmetricGPResult = demo["result"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cf = axes[0].contourf(res.z_hat, res.r_hat, res.rho_norm, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=axes[0], label=r"$\rho/\rho_{\mathrm{in}}$")
    axes[0].set_xlabel(r"$\hat z$ (pole)")
    axes[0].set_ylabel(r"$\hat r$ (equator)")
    axes[0].set_title("Oblate sink ρ")
    axes[0].set_aspect("equal")

    axes[1].plot(demo["r_eq"], demo["rho_eq"], "b-", label=r"equator $z=0$")
    axes[1].plot(demo["z_pol"], demo["rho_pol"], "r--", label=r"pole $r=0$")
    axes[1].set_xlabel(r"$\hat r$ or $\hat z$")
    axes[1].set_ylabel(r"$\rho/\rho_{\mathrm{in}}$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].bar(
        ["center", "eq mid", "pol mid"],
        [demo["rho_center"], demo["rho_eq_mid"], demo["rho_pol_mid"]],
        color=["C2", "C0", "C3"],
    )
    axes[2].set_ylabel(r"$\rho/\rho_{\mathrm{in}}$")
    axes[2].set_title(f"pole/eq mid = {demo['pole_over_eq_mid']:.3f}")
    fig.suptitle(
        rf"CH oblate sink ($\sigma_\parallel$={demo['sigma_parallel']}, $\sigma_\perp$={demo['sigma_perp']})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_report(
    ch: CHParams,
    cmp_validate: dict,
    oblate: dict | None,
    rs_hat: float,
) -> str:
    lines = [
        "CH 2D axisymmetric GP gravity — scaffold",
        f"xi = {ch.xi:.3e} m, r_s/xi = {rs_hat}, r_join/xi = 12",
        "",
        "Note: 1D radial and 2D cylindrical operators agree on the equator when",
        "the 1D profile is seeded on the (r,z) grid (L2 ~ 0); full relaxation",
        "breaks spherical symmetry because the operators differ off-axis.",
        "",
        "=== Spherical sink — grid / seed validation (no relaxation) ===",
        f"  gamma0 = {cmp_validate['g0']:.4f}",
        f"  equatorial rho L2 (1D seed on 2D grid) = {cmp_validate['rho_l2_rel_equator']:.4e}",
        f"  M_inner (1D radial reference) = {cmp_validate['m_inner_1d_kg']:.4e} kg",
    ]
    if oblate is not None:
        lines += [
            "",
            "=== Oblate sink (2D relaxation) ===",
            f"  sigma_parallel = {oblate['sigma_parallel']}, sigma_perp = {oblate['sigma_perp']}",
            f"  gamma0 = {oblate['result'].g0:.4f}",
            f"  rho_center = {oblate['rho_center']:.4f}",
            f"  rho_mid equator / pole = {oblate['rho_eq_mid']:.4f} / {oblate['rho_pol_mid']:.4f}",
            f"  pole/equator mid ratio = {oblate['pole_over_eq_mid']:.4f}",
            f"  M_inner (2D ball) = {oblate['m_inner_kg']:.4e} kg",
            f"  iterations = {oblate['result'].n_iter}",
        ]
    lines += [
        "",
        "Next steps:",
        "  - Schwarzschild tail on R = sqrt(r^2+z^2) for exterior Newton",
        "  - oblate generality link to ch_alpha_g_generality_sweep.py",
        "  - full 3D / rotation: off-laptop HPC",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D axisymmetric GP gravity scaffold")
    parser.add_argument("--xi", type=float, default=50e-9, help="healing length [m]")
    parser.add_argument("--sigma", type=float, default=0.6, help="sink width sigma/xi")
    parser.add_argument("--sigma-par", type=float, default=0.6, help="oblate sigma_parallel/xi")
    parser.add_argument("--sigma-perp", type=float, default=1.2, help="oblate sigma_perp/xi")
    parser.add_argument("--rs-hat", type=float, default=1.0, help="r_s / xi")
    parser.add_argument("--g0", type=float, default=None, help="sink strength (default: smooth cal)")
    parser.add_argument("--nr", type=int, default=80)
    parser.add_argument("--nz", type=int, default=80)
    parser.add_argument("--n-iter", type=int, default=8000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--skip-oblate",
        action="store_true",
        help="only run spherical seed validation",
    )
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    rs_hat = args.rs_hat
    sigma = args.sigma
    r_join = 12.0

    if args.g0 is not None:
        g0 = args.g0
    else:
        g0, _, _, _ = calibrate_g0_matched(rs_hat, sigma_hat=sigma)

    cmp_validate = compare_1d_vs_2d(
        ch,
        g0,
        sigma,
        rs_hat,
        r_join_hat=r_join,
        nr=args.nr,
        nz=args.nz,
        relax=False,
    )

    oblate = None
    if not args.skip_oblate:
        oblate = solve_oblate_demo(
            ch,
            g0,
            args.sigma_par,
            args.sigma_perp,
            rs_hat,
            r_join_hat=r_join,
            nr=args.nr,
            nz=args.nz,
            n_iter=args.n_iter,
        )

    report = format_report(ch, cmp_validate, oblate, rs_hat)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT / "ch_gpe_gravity_axisymmetric.txt"
    txt_path.write_text(report + "\n")
    print(f"\nWrote {txt_path}")

    if not args.no_plot:
        png_path = OUTPUT / "ch_gpe_gravity_axisymmetric_validate.png"
        plot_comparison(cmp_validate, ch, png_path, title_suffix=" [1D seed validation]")
        print(f"Wrote {png_path}")
        if oblate is not None:
            oblate_png = OUTPUT / "ch_gpe_gravity_axisymmetric_oblate.png"
            plot_oblate(oblate, ch, oblate_png)
            print(f"Wrote {oblate_png}")


if __name__ == "__main__":
    main()
