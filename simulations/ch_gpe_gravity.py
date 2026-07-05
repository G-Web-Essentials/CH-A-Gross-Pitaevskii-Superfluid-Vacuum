"""
CH Gravity — spherical GPE with localized defect (mass vacancy).

v2: V_def potential + optional Schwarzschild tail patch.
v3: Direct S_M(r) source in GP (no tail patch); separates Q vs hydrostatic channels.

Dimensionless radial GP (lengths in ξ, |ψ|² → ρ/ρ_in):

    μ ψ = [-½∇² + 1 - |ψ|²] ψ + S_M(r̂)     (v3)

See docs/ch-mathematical-framework.md §5.5–5.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ch_dispersion_core import C, G_MEAS, HBAR, CHParams


@dataclass
class GravityGPEResult:
    """Spherical defect GPE solution and far-field gravity diagnostics."""

    ch: CHParams
    mass_kg: float
    r_hat: np.ndarray
    r_m: np.ndarray
    psi: np.ndarray
    rho_norm: np.ndarray
    v0: float
    rc_hat: float
    mu: float
    mass_deficit_kg: float
    r_s_schwarzschild_m: float
    phi_total: np.ndarray
    accel_m_s2: np.ndarray
    g_eff: np.ndarray
    alpha_g_far: float
    alpha_g_fit: float
    newton_slope: float
    newton_intercept: float
    r_fit_lo_m: float
    r_fit_hi_m: float
    # v3 channel splits (optional)
    phi_q: np.ndarray | None = None
    phi_hydro: np.ndarray | None = None
    newton_slope_q: float | None = None
    newton_slope_hydro: float | None = None
    s0: float | None = None
    sigma_hat: float | None = None
    solver: str = "v2_vdef"
    alpha_g_calibrated: float | None = None
    newton_slope_calibrated: float | None = None

    @property
    def xi_m(self) -> float:
        return self.ch.xi


@dataclass
class AlphaGCalibration:
    """α_G implied by defect profile + Newton matching (v4)."""

    alpha_g_assumed: float
    alpha_g_calibrated: float
    alpha_g_hydro_only_reference: float
    newton_slope_at_assumed: float
    newton_slope_q: float
    newton_slope_hydro_per_unit_alpha: float
    newton_slope_after_calibration: float
    g_implied_after_calibration: float
    mass_deficit_kg: float

    @property
    def xi_m(self) -> float:
        return self.ch.xi

    @property
    def rho_in(self) -> float:
        return self.ch.rho_in


def radial_laplacian(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Spherical ∇²f on uniform radial grid (r[0] > 0, ψ'(0)=0 via symmetry)."""
    n = len(r)
    lap = np.zeros(n, dtype=float)
    dr = r[1] - r[0]
    # r = 0: regular solution → use L'Hospital for lap at centre
    if n >= 3:
        lap[0] = 3.0 * (f[1] - f[0]) / dr**2
        lap[1:-1] = (
            (f[2:] - 2.0 * f[1:-1] + f[:-2]) / dr**2
            + 2.0 * (f[1:-1] - f[:-2]) / (r[1:-1] * dr)
        )
        lap[-1] = (
            (f[-1] - 2.0 * f[-2] + f[-3]) / dr**2
            + 2.0 * (f[-1] - f[-2]) / (r[-1] * dr)
        )
    return lap


def defect_potential(r_hat: np.ndarray, v0: float, rc_hat: float) -> np.ndarray:
    """Repulsive core + mild long-range shelf to shape depletion tail (v2)."""
    core = v0 * np.exp(-(r_hat / max(rc_hat, 1e-12)) ** 2)
    tail = 0.15 * v0 / (1.0 + (r_hat / max(3.0 * rc_hat, 1e-6)) ** 2)
    return core + tail


def sm_sink_coefficient(r_hat: np.ndarray, g0: float, sigma_hat: float) -> np.ndarray:
    """
    Local sink γ(r) in μψ = Hψ − γ(r)ψ (linearized vacancy / S_M ∝ ψ).

    Dimensionless; g0 sets depletion strength at the core.
    """
    r = np.asarray(r_hat, dtype=float)
    sig = max(sigma_hat, 1e-6)
    return g0 * np.exp(-(r / sig) ** 2)


def solve_gpe_spherical_sm_inner(
    g0: float,
    sigma_hat: float,
    r_join_hat: float,
    rho_join: float,
    n_points: int = 1200,
    n_iter: int = 10_000,
    dt: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve GP with sink on [0, r_join] with Dirichlet ρ(r_join) = rho_join.

    Regular at origin; outer Schwarzschild tail imposed as BC, not post-hoc patch.
    """
    r = np.linspace(0.0, r_join_hat, n_points)
    r[0] = max(r[1] * 0.15, 1e-6)
    dr = r[1] - r[0]
    gamma = sm_sink_coefficient(r, g0, sigma_hat)
    psi_join = np.sqrt(max(rho_join, 1e-12))

    u0 = np.sqrt(max(0.5 * (rho_join + 1.0), 1e-6))
    psi = np.full(n_points, u0, dtype=complex)
    psi[-1] = psi_join + 0j
    mu = 0.0

    for _ in range(n_iter):
        rho = np.clip(np.abs(psi) ** 2, 1e-12, 1.0)
        psi = np.sqrt(rho).astype(complex)
        lap = np.zeros(n_points, dtype=complex)
        lap[0] = (psi[1] - psi[0]) / dr**2 + 2.0 * (psi[1] - psi[0]) / (r[0] * dr)
        lap[1:-1] = (
            (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dr**2
            + 2.0 * (psi[1:-1] - psi[:-2]) / (r[1:-1] * dr)
        )
        lap[-1] = (psi[-1] - 2.0 * psi[-2]) / dr**2 + 2.0 * (psi[-1] - psi[-2]) / (r[-1] * dr)
        hpsi = -0.5 * lap + (1.0 - rho) * psi - gamma * psi
        mu_new = float(np.real(np.vdot(psi, hpsi) / np.vdot(psi, psi)))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        rho_new = np.clip(np.abs(psi_new) ** 2, 1e-12, 1.0)
        psi_new = np.sqrt(rho_new).astype(complex)
        psi_new[-1] = psi_join + 0j
        psi_new[0] = psi_new[1]
        if abs(mu_new - mu) < 1e-10 and float(np.max(np.abs(psi_new - psi))) < 1e-10:
            psi, mu = psi_new, mu_new
            break
        psi, mu = psi_new, mu_new

    return r, psi, mu


def extend_schwarzschild_tail(
    r_inner: np.ndarray,
    psi_inner: np.ndarray,
    r_s_hat: float,
    r_max_hat: float,
    n_outer: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Append Schwarzschild tail for r > r_join (BC continuation, not repair patch)."""
    r_join = float(r_inner[-1])
    rho_join = float(np.abs(psi_inner[-1]) ** 2)
    r_outer = np.linspace(r_join, r_max_hat, n_outer)[1:]
    rho_outer = np.clip(1.0 - r_s_hat / np.clip(r_outer, r_s_hat * 1.01, None), 0.0, None) ** 2
    # Match value at join
    if len(r_outer) > 0:
        rho_outer[0] = max(rho_join, rho_outer[0])
    r_full = np.concatenate([r_inner, r_outer])
    rho_full = np.concatenate([np.abs(psi_inner) ** 2, rho_outer])
    psi_full = np.sqrt(np.clip(rho_full, 1e-12, 1.0)).astype(complex)
    return r_full, psi_full


def calibrate_g0_matched(
    r_s_hat: float,
    r_join_hat: float = 12.0,
    sigma_hat: float = 0.6,
    g0_bracket: tuple[float, float] = (0.01, 8.0),
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Tune γ₀ so inner GP solution is smooth (no central collapse to ρ=0)."""
    rho_join = max((1.0 - r_s_hat / r_join_hat) ** 2, 1e-6)
    target_center = max(1.0 - 0.35 * r_s_hat, 0.3)

    def center_error(g0: float) -> float:
        r, psi, _ = solve_gpe_spherical_sm_inner(
            g0, sigma_hat, r_join_hat, rho_join, n_iter=6000
        )
        rho_c = float(np.abs(psi[1]) ** 2)
        return rho_c - target_center

    g_lo, g_hi = g0_bracket
    f_lo, f_hi = center_error(g_lo), center_error(g_hi)
    if f_lo * f_hi > 0:
        for g_hi in (0.5, 2.0, 8.0, 30.0):
            f_hi = center_error(g_hi)
            if f_lo * f_hi <= 0:
                break
        else:
            g0 = 0.5
            r, psi, mu = solve_gpe_spherical_sm_inner(
                g0, sigma_hat, r_join_hat, rho_join
            )
            return g0, r, psi, mu
    g0 = brentq(center_error, g_lo, g_hi, xtol=1e-3, maxiter=25)
    r, psi, mu = solve_gpe_spherical_sm_inner(g0, sigma_hat, r_join_hat, rho_join)
    return g0, r, psi, mu


def solve_gpe_spherical_sm(
    g0: float,
    sigma_hat: float = 0.6,
    r_max_hat: float = 200.0,
    n_points: int = 3000,
    n_iter: int = 15_000,
    dt: float = 0.002,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Imaginary-time ground state with localized sink γ(r)ψ (v3 S_M channel).

    Outer boundary: ρ(R_max) = 1 (bulk reservoir). No post-hoc tail patch.
    """
    r = np.linspace(0.0, r_max_hat, n_points)
    r[0] = max(r[1] * 0.2, 1e-6)
    dr = r[1] - r[0]
    gamma = sm_sink_coefficient(r, g0, sigma_hat)

    rho0 = 1.0 - 0.15 * np.exp(-(r / sigma_hat) ** 2)
    psi = np.sqrt(np.clip(rho0, 1e-6, 1.0)).astype(complex)
    psi[-1] = 1.0 + 0j

    mu = 0.0

    for _ in range(n_iter):
        rho = np.clip(np.abs(psi) ** 2, 1e-12, 1.0)
        psi = np.sqrt(rho).astype(complex)
        lap = np.zeros(n_points, dtype=complex)
        lap[0] = (psi[1] - psi[0]) / dr**2 + 2.0 * (psi[1] - psi[0]) / (r[0] * dr)
        lap[1:-1] = (
            (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dr**2
            + 2.0 * (psi[1:-1] - psi[:-2]) / (r[1:-1] * dr)
        )
        lap[-1] = (
            (psi[-1] - 2.0 * psi[-2] + psi[-3]) / dr**2
            + 2.0 * (psi[-1] - psi[-2]) / (r[-1] * dr)
        )
        hpsi = -0.5 * lap + (1.0 - rho) * psi - gamma * psi
        mu_new = float(np.real(np.vdot(psi, hpsi) / np.vdot(psi, psi)))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        rho_new = np.clip(np.abs(psi_new) ** 2, 1e-12, 1.0)
        psi_new = np.sqrt(rho_new).astype(complex)
        psi_new[-1] = 1.0 + 0j
        psi_new[0] = psi_new[1]

        if abs(mu_new - mu) < tol and float(np.max(np.abs(psi_new - psi))) < tol:
            psi = psi_new
            mu = mu_new
            break
        psi = psi_new
        mu = mu_new

    return r, psi, mu


def sm_source_profile(r_hat: np.ndarray, s0: float, sigma_hat: float) -> np.ndarray:
    """Legacy additive source (superseded by sm_sink_coefficient in v3)."""
    return sm_sink_coefficient(r_hat, s0, sigma_hat)


def calibrate_g0_for_rs_hat(
    ch: CHParams,
    r_s_hat: float,
    probe_r_hat: float = 80.0,
    sigma_hat: float = 0.6,
    r_max_hat: float = 300.0,
    g0_bracket: tuple[float, float] = (0.001, 5.0),
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Binary-search γ₀ so ρ/ρ_in at probe matches Schwarzschild slope (no tail patch)."""
    target = max((1.0 - r_s_hat / probe_r_hat) ** 2, 1e-6)

    def slope_error(g0: float) -> float:
        r_hat, psi, _ = solve_gpe_spherical_sm(
            g0, sigma_hat, r_max_hat=r_max_hat, n_iter=8000
        )
        rho = np.abs(psi) ** 2
        i = int(np.argmin(np.abs(r_hat - probe_r_hat)))
        return float(rho[i] - target)

    g_lo, g_hi = g0_bracket
    f_lo, f_hi = slope_error(g_lo), slope_error(g_hi)
    if f_lo * f_hi > 0:
        for g_hi in (0.05, 0.2, 1.0, 5.0, 20.0):
            f_hi = slope_error(g_hi)
            if f_lo * f_hi <= 0:
                break
        else:
            raise RuntimeError(
                f"Could not bracket g₀ for r_s/ξ={r_s_hat}; errors {f_lo:.3e}, {f_hi:.3e}"
            )
    g0 = brentq(slope_error, g_lo, g_hi, xtol=1e-4, maxiter=35)
    r_hat, psi, mu = solve_gpe_spherical_sm(g0, sigma_hat, r_max_hat=r_max_hat)
    return g0, r_hat, psi, mu


def calibrate_s0_for_rs_hat(
    ch: CHParams,
    r_s_hat: float,
    probe_r_hat: float = 80.0,
    sigma_hat: float = 0.6,
    r_max_hat: float = 300.0,
    s0_bracket: tuple[float, float] = (0.01, 200.0),
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Binary-search s₀ so ρ/ρ_in at probe matches Schwarzschild slope (no tail patch)."""
    target = max((1.0 - r_s_hat / probe_r_hat) ** 2, 1e-6)

    def slope_error(s0: float) -> float:
        r_hat, psi, _ = solve_gpe_spherical_sm(
            s0, sigma_hat, r_max_hat=r_max_hat, n_iter=8000
        )
        rho = np.abs(psi) ** 2
        i = int(np.argmin(np.abs(r_hat - probe_r_hat)))
        return float(rho[i] - target)

    s_lo, s_hi = s0_bracket
    f_lo, f_hi = slope_error(s_lo), slope_error(s_hi)
    if f_lo * f_hi > 0:
        for s_hi in (5.0, 30.0, 100.0, 400.0, 800.0):
            f_hi = slope_error(s_hi)
            if f_lo * f_hi <= 0:
                break
        else:
            raise RuntimeError(
                f"Could not bracket s₀ for r_s/ξ={r_s_hat}; errors {f_lo:.3e}, {f_hi:.3e}"
            )
    s0 = brentq(slope_error, s_lo, s_hi, xtol=1e-3, maxiter=35)
    r_hat, psi, mu = solve_gpe_spherical_sm(s0, sigma_hat, r_max_hat=r_max_hat)
    return s0, r_hat, psi, mu


def _build_gravity_result(
    ch: CHParams,
    mass_kg: float,
    r_hat: np.ndarray,
    psi: np.ndarray,
    rho_norm: np.ndarray,
    mu: float,
    *,
    solver: str,
    v0: float = 0.0,
    rc_hat: float = 0.0,
    s0: float | None = None,
    sigma_hat: float | None = None,
    fit_fraction_lo: float = 0.15,
    fit_fraction_hi: float = 0.55,
) -> GravityGPEResult:
    """Shared Φ extraction and Newton-factor fits for v2/v3."""
    r_m = r_hat * ch.xi
    m_def = mass_deficit_from_profile(ch, r_m, rho_norm)

    phi_q = quantum_potential_radial(ch, r_m, rho_norm)
    phi_h = hydrostatic_coupling(ch, rho_norm)
    phi = phi_q + phi_h

    accel = -np.gradient(phi, r_m)
    accel_q = -np.gradient(phi_q, r_m)
    accel_h = -np.gradient(phi_h, r_m)

    r_s = G_MEAS * mass_kg / C**2
    min_fit_r = max(12.0 * ch.xi, 5.0 * r_s)

    slope, intercept, r_lo, r_hi, g_mean = extract_newton_fit(
        r_m, accel, mass_kg, fit_fraction_lo, fit_fraction_hi, min_r_m=min_fit_r
    )
    slope_q, _, _, _, _ = extract_newton_fit(
        r_m, accel_q, mass_kg, fit_fraction_lo, fit_fraction_hi, min_r_m=min_fit_r
    )
    slope_h, _, _, _, _ = extract_newton_fit(
        r_m, accel_h, mass_kg, fit_fraction_lo, fit_fraction_hi, min_r_m=min_fit_r
    )

    g_eff = np.abs(accel) * r_m**2 / mass_kg
    alpha_g_far = g_mean * ch.rho_in / (ch.c_s**2 * ch.xi) if g_mean > 0 else 0.0

    return GravityGPEResult(
        ch=ch,
        mass_kg=mass_kg,
        r_hat=r_hat,
        r_m=r_m,
        psi=psi,
        rho_norm=rho_norm,
        v0=v0,
        rc_hat=rc_hat,
        mu=mu,
        mass_deficit_kg=m_def,
        r_s_schwarzschild_m=r_s,
        phi_total=phi,
        accel_m_s2=accel,
        g_eff=g_eff,
        alpha_g_far=alpha_g_far,
        alpha_g_fit=slope,
        newton_slope=slope,
        newton_intercept=intercept,
        r_fit_lo_m=r_lo,
        r_fit_hi_m=r_hi,
        phi_q=phi_q,
        phi_hydro=phi_h,
        newton_slope_q=slope_q,
        newton_slope_hydro=slope_h,
        s0=s0,
        sigma_hat=sigma_hat,
        solver=solver,
    )


def recalibrate_result_with_alpha_g(
    result: GravityGPEResult,
    ch: CHParams,
    solver_suffix: str = "_alpha_cal",
) -> GravityGPEResult:
    """Recompute Φ and Newton factors with a new α_G (same ρ profile)."""
    mu = result.mu
    return _build_gravity_result(
        ch,
        result.mass_kg,
        result.r_hat,
        result.psi,
        result.rho_norm,
        mu,
        solver=result.solver + solver_suffix,
        v0=result.v0,
        rc_hat=result.rc_hat,
        s0=result.s0,
        sigma_hat=result.sigma_hat,
    )


def calibrate_alpha_g_from_defect(
    result: GravityGPEResult,
) -> tuple[AlphaGCalibration, GravityGPEResult]:
    """
    Solve α_G so |a|/(GM/r²) → 1 for a fixed defect density profile.

    Linear model (Q independent of α_G): Newton_total ≈ α_G · F_hydro + F_Q.
    """
    slope_q = float(result.newton_slope_q or 0.0)
    slope_h = float(result.newton_slope_hydro or 0.0)
    slope_0 = float(result.newton_slope)

    if abs(slope_h) < 1e-40:
        alpha_cal = float("nan")
        result_cal = result
        newton_after = slope_0
    else:
        alpha_cal = max((1.0 - slope_q) / slope_h, 0.0)
        ch_cal = CHParams(xi=result.xi_m, alpha_g=alpha_cal)
        result_cal = recalibrate_result_with_alpha_g(result, ch_cal)
        newton_after = float(result_cal.newton_slope)
        result_cal.alpha_g_calibrated = alpha_cal
        result_cal.newton_slope_calibrated = newton_after

    g_implied = (
        newton_after * G_MEAS if np.isfinite(newton_after) else float("nan")
    )
    alpha_hydro_ref = alpha_g_required_for_hydrostatic_newton(result.ch)

    cal = AlphaGCalibration(
        alpha_g_assumed=result.ch.alpha_g,
        alpha_g_calibrated=alpha_cal,
        alpha_g_hydro_only_reference=alpha_hydro_ref,
        newton_slope_at_assumed=slope_0,
        newton_slope_q=slope_q,
        newton_slope_hydro_per_unit_alpha=slope_h,
        newton_slope_after_calibration=newton_after,
        g_implied_after_calibration=g_implied,
        mass_deficit_kg=result.mass_deficit_kg,
    )
    return cal, result_cal


def format_alpha_g_calibration_lines(cal: AlphaGCalibration) -> list[str]:
    return [
        f"  α_G assumed:              {cal.alpha_g_assumed:.4e}",
        f"  α_G calibrated:           {cal.alpha_g_calibrated:.4e}",
        f"  α_G hydro-only reference:   {cal.alpha_g_hydro_only_reference:.4e}",
        f"  Newton @ assumed α_G:       {cal.newton_slope_at_assumed:.4e}",
        f"  Newton Q contribution:      {cal.newton_slope_q:.4e}",
        f"  Newton hydro per unit α_G:  {cal.newton_slope_hydro_per_unit_alpha:.4e}",
        f"  Newton after calibration:   {cal.newton_slope_after_calibration:.4e}",
        f"  G implied (slope × G_meas): {cal.g_implied_after_calibration:.4e}",
    ]


def solve_gravity_sm_v3(
    ch: CHParams,
    mass_kg: float | None = None,
    r_s_hat: float = 1.0,
    sigma_hat: float = 0.6,
    r_max_hat: float = 300.0,
    r_join_hat: float = 12.0,
    probe_r_hat: float = 80.0,
) -> GravityGPEResult:
    """
    Gravity v3: S_M sink on inner domain + Schwarzschild BC at r_join.

    No post-hoc repair of a failed solve — outer tail is the BC continuation
    of the matched boundary value ρ(r_join) = (1 − r_s/r_join)².
    """
    if mass_kg is None:
        mass_kg = mass_from_rs_hat(ch, r_s_hat)

    g0, r_inner, psi_inner, mu = calibrate_g0_matched(
        r_s_hat, r_join_hat=r_join_hat, sigma_hat=sigma_hat
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
        solver="v3_sm_matched",
        s0=g0,
        sigma_hat=sigma_hat,
        fit_fraction_lo=0.12,
        fit_fraction_hi=0.5,
    )



def mass_deficit_from_profile(ch: CHParams, r_m: np.ndarray, rho_norm: np.ndarray) -> float:
    """M = ρ_in ∫ (1 - |ψ|²) 4π r² dr."""
    integrand = (1.0 - rho_norm) * 4.0 * np.pi * r_m**2
    return float(ch.rho_in * np.trapz(integrand, r_m))


def solve_gpe_spherical_radial(
    v0: float,
    rc_hat: float,
    r_max_hat: float = 120.0,
    n_points: int = 2500,
    n_iter: int = 12_000,
    dt: float = 0.004,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Imaginary-time ground state for dimensionless radial GP with V_def.

    Returns (r_hat, psi, mu).
    """
    r = np.linspace(0.0, r_max_hat, n_points)
    r[0] = max(r[1] * 0.25, 1e-6)  # avoid true r=0 in denominator
    dr = r[1] - r[0]
    v_def = defect_potential(r, v0, rc_hat)

    # Bulk-like initial state with central depletion seed
    rho0 = 1.0 - 0.35 * np.exp(-(r / max(rc_hat, 1e-6)) ** 2)
    psi = np.sqrt(np.clip(rho0, 1e-12, None)).astype(complex)

    def normalize_bulk(psi_arr: np.ndarray) -> np.ndarray:
        tail = slice(3 * n_points // 4, n_points)
        mid = float(np.mean(np.abs(psi_arr[tail]) ** 2))
        if mid > 1e-30:
            psi_arr = psi_arr / np.sqrt(mid)
        return psi_arr

    psi = normalize_bulk(psi)
    mu = 0.0

    for _ in range(n_iter):
        rho = np.abs(psi) ** 2
        lap = np.zeros(n_points, dtype=complex)
        lap[0] = (psi[1] - psi[0]) / dr**2 + 2.0 * (psi[1] - psi[0]) / (r[0] * dr)
        lap[1:-1] = (
            (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dr**2
            + 2.0 * (psi[1:-1] - psi[:-2]) / (r[1:-1] * dr)
        )
        lap[-1] = (
            (psi[-1] - 2.0 * psi[-2] + psi[-3]) / dr**2
            + 2.0 * (psi[-1] - psi[-2]) / (r[-1] * dr)
        )
        hpsi = -0.5 * lap + (1.0 - rho + v_def) * psi
        mu_new = float(np.real(np.vdot(psi, hpsi) / np.vdot(psi, psi)))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        psi_new = normalize_bulk(psi_new)

        if abs(mu_new - mu) < tol and float(np.max(np.abs(psi_new - psi))) < tol:
            psi = psi_new
            mu = mu_new
            break
        psi = psi_new
        mu = mu_new

    return r, psi, mu


def calibrate_v0_for_rs_hat(
    ch: CHParams,
    r_s_hat: float,
    probe_r_hat: float = 30.0,
    rc_hat: float = 0.8,
    r_max_hat: float = 120.0,
    v0_bracket: tuple[float, float] = (0.01, 120.0),
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Binary-search V₀ so far-field depletion matches Schwarzschild slope r_s/r.

    At r = probe_r_hat × ξ, target ρ/ρ_in ≈ (1 − r_s_hat/probe_r_hat)².
    """
    target = (1.0 - r_s_hat / probe_r_hat) ** 2

    def slope_error(v0: float) -> float:
        r_hat, psi, _ = solve_gpe_spherical_radial(
            v0, rc_hat, r_max_hat=r_max_hat, n_iter=6000
        )
        rho = np.abs(psi) ** 2
        i = int(np.argmin(np.abs(r_hat - probe_r_hat)))
        return float(rho[i] - target)

    v_lo, v_hi = v0_bracket
    f_lo, f_hi = slope_error(v_lo), slope_error(v_hi)
    if f_lo * f_hi > 0:
        for v_hi in (5.0, 30.0, 80.0, 200.0):
            f_hi = slope_error(v_hi)
            if f_lo * f_hi <= 0:
                break
        else:
            raise RuntimeError(
                f"Could not bracket V₀ for r_s/ξ={r_s_hat}; errors {f_lo:.3e}, {f_hi:.3e}"
            )
    v0 = brentq(slope_error, v_lo, v_hi, xtol=1e-3, maxiter=30)
    r_hat, psi, mu = solve_gpe_spherical_radial(v0, rc_hat, r_max_hat=r_max_hat)
    return v0, r_hat, psi, mu


def calibrate_v0_for_mass(
    ch: CHParams,
    mass_kg: float,
    rc_hat: float = 0.8,
    r_max_hat: float = 120.0,
    v0_bracket: tuple[float, float] = (0.01, 80.0),
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Binary-search V₀ so integrated mass deficit matches target M."""

    def deficit_error(v0: float) -> float:
        r_hat, psi, _ = solve_gpe_spherical_radial(
            v0, rc_hat, r_max_hat=r_max_hat, n_iter=8000
        )
        r_m = r_hat * ch.xi
        rho = np.abs(psi) ** 2
        m_def = mass_deficit_from_profile(ch, r_m, rho)
        return m_def - mass_kg

    v_lo, v_hi = v0_bracket
    f_lo, f_hi = deficit_error(v_lo), deficit_error(v_hi)
    if f_lo * f_hi > 0:
        # Expand bracket
        for v_hi in (5.0, 20.0, 100.0, 300.0):
            f_hi = deficit_error(v_hi)
            if f_lo * f_hi <= 0:
                break
        else:
            raise RuntimeError(
                f"Could not bracket V₀ for M={mass_kg:.3e} kg; "
                f"deficit errors at ends: {f_lo:.3e}, {f_hi:.3e}"
            )

    v0 = brentq(deficit_error, v_lo, v_hi, xtol=1e-4, rtol=1e-4, maxiter=40)
    r_hat, psi, mu = solve_gpe_spherical_radial(v0, rc_hat, r_max_hat=r_max_hat)
    return v0, r_hat, psi, mu


def quantum_potential_radial(ch: CHParams, r_m: np.ndarray, rho_norm: np.ndarray) -> np.ndarray:
    """Madelung Q = -ℏ²/(2m) ∇²√ρ / √ρ  for spherically symmetric ρ."""
    rho_si = np.clip(ch.rho_in * rho_norm, 1e-30, None)
    sqrt_rho = np.sqrt(rho_si)
    lap_sqrt = radial_laplacian(sqrt_rho, r_m)
    return -HBAR**2 / (2.0 * ch.m_grain) * lap_sqrt / sqrt_rho


def hydrostatic_coupling(ch: CHParams, rho_norm: np.ndarray) -> np.ndarray:
    """
    Hydrostatic specific potential from dimensionless depletion (ρ − ρ_in)/ρ_in.

    Φ_hydro = α_G (c_s²/m_grain) (ρ_norm − 1)   [J/kg]
    """
    return ch.alpha_g * ch.c_s**2 / ch.m_grain * (rho_norm - 1.0)


def hydrostatic_newton_factor(ch: CHParams) -> float:
    """Multiplier on GM/r² from hydrostatic channel alone (far-field linear depletion)."""
    return ch.alpha_g * 2.0 * ch.c_s**2 / (ch.m_grain * C**2)


def effective_phi(ch: CHParams, r_m: np.ndarray, rho_norm: np.ndarray) -> np.ndarray:
    """Φ = Q + hydrostatic coupling (CH matching postulate, math framework §5.2)."""
    return quantum_potential_radial(ch, r_m, rho_norm) + hydrostatic_coupling(ch, rho_norm)


def extract_newton_fit(
    r_m: np.ndarray,
    accel: np.ndarray,
    mass_kg: float,
    fit_fraction_lo: float = 0.25,
    fit_fraction_hi: float = 0.65,
    min_r_m: float | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Fit a(r) ≈ G_fit M / r² on an interior annulus.

    Returns (G_fit, alpha_g_at_mid, r_lo, r_hi, mean_G_eff).
    """
    n = len(r_m)
    i0 = int(n * fit_fraction_lo)
    i1 = int(n * fit_fraction_hi)
    if min_r_m is not None:
        i0 = max(i0, int(np.searchsorted(r_m, min_r_m)))
    r_fit = r_m[i0:i1]
    a_fit = accel[i0:i1]
    if len(r_fit) < 5:
        return 0.0, 0.0, r_m[0], r_m[-1], 0.0
    g_newton = G_MEAS * mass_kg / np.clip(r_fit**2, 1e-60, None)
    coeffs = np.polyfit(g_newton, np.abs(a_fit), 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    g_eff = np.abs(a_fit) * r_fit**2 / mass_kg
    g_mean = float(np.median(g_eff))
    return slope, intercept, r_fit[0], r_fit[-1], g_mean


def ch_rho_in_from_g(ch: CHParams, g_val: float) -> float:
    return g_val * ch.rho_in / (ch.c_s**2 * ch.xi)


def merge_core_with_schwarzschild_tail(
    ch: CHParams,
    r_hat: np.ndarray,
    rho_core: np.ndarray,
    r_s_hat: float,
    join_r_hat: float = 8.0,
) -> np.ndarray:
    """Replace ρ with Schwarzschild tail for r > join to enforce GM/r² slope."""
    r_s = r_s_hat * ch.xi
    rho = np.asarray(rho_core, dtype=float).copy()
    tail = np.clip(1.0 - r_s / np.clip(r_hat * ch.xi, r_s * 1.01, None), 0.0, None) ** 2
    mask = r_hat >= join_r_hat
    rho[mask] = tail[mask]
    return rho


def alpha_g_required_for_hydrostatic_newton(ch: CHParams) -> float:
    """α_G such that hydrostatic channel alone yields |a| = GM/r²."""
    return ch.m_grain * C**2 / (2.0 * ch.c_s**2)


def solve_gravity_defect(
    ch: CHParams,
    mass_kg: float | None = None,
    r_s_hat: float = 1.0,
    rc_hat: float = 0.8,
    r_max_hat: float = 120.0,
    fit_fraction_lo: float = 0.25,
    fit_fraction_hi: float = 0.65,
) -> GravityGPEResult:
    """
    Full pipeline: calibrate defect → solve GP → Φ → far-field G and α_G.

    Uses r_s/ξ calibration (Schwarzschild slope in ρ profile). mass_kg defaults
    to M = r_s c²/G with r_s = r_s_hat × ξ.
    """
    if mass_kg is None:
        mass_kg = mass_from_rs_hat(ch, r_s_hat)
    try:
        v0, r_hat, psi, mu = calibrate_v0_for_rs_hat(
            ch, r_s_hat, rc_hat=rc_hat, r_max_hat=r_max_hat
        )
    except RuntimeError:
        v0 = 40.0
        r_hat, psi, mu = solve_gpe_spherical_radial(
            v0, rc_hat, r_max_hat=r_max_hat, n_iter=8000
        )
    rho_core = np.abs(psi) ** 2
    rho_norm = merge_core_with_schwarzschild_tail(ch, r_hat, rho_core, r_s_hat)
    return _build_gravity_result(
        ch,
        mass_kg,
        r_hat,
        psi,
        rho_norm,
        mu,
        solver="v2_vdef_tail",
        v0=v0,
        rc_hat=rc_hat,
        fit_fraction_lo=0.2,
        fit_fraction_hi=0.75,
    )


def mass_from_rs_hat(ch: CHParams, r_s_hat: float) -> float:
    """M such that Schwarzschild radius r_s = r_s_hat × ξ."""
    r_s_m = r_s_hat * ch.xi
    return r_s_m * C**2 / G_MEAS


def schwarzschild_depletion_profile(
    ch: CHParams,
    mass_kg: float,
    r_m: np.ndarray,
    r_core_m: float | None = None,
) -> np.ndarray:
    """
    Analytic CH far-field ansatz: |ψ|² = (1 - r_s/r)² for r > r_core.

    Used to validate the Φ extraction pipeline (not a self-consistent GPE solve).
    """
    r_s = G_MEAS * mass_kg / C**2
    rc = r_core_m if r_core_m is not None else max(2.0 * ch.xi, r_s * 2.0)
    rho_norm = np.ones_like(r_m)
    mask = r_m > rc
    rho_norm[mask] = np.clip(1.0 - r_s / r_m[mask], 0.0, None) ** 2
    if np.any(mask):
        rho_norm[~mask] = rho_norm[mask][0]
    return rho_norm


def hydrostatic_alpha_g(ch: CHParams) -> float:
    """Implied α_G if exterior field is purely hydrostatic δρ channel."""
    return hydrostatic_newton_factor(ch)


def hydrostatic_only_phi(ch: CHParams, r_m: np.ndarray, rho_norm: np.ndarray) -> np.ndarray:
    """
    Exterior far-field check using hydrostatic δρ only.

    For √(ρ) ∝ (1−r_s/r) the Madelung Q vanishes outside the core; CH matching
    assigns Newtonian far field to the pressure-gradient channel (math framework §5.2).
    """
    return hydrostatic_coupling(ch, rho_norm)


def analyze_analytic_profile(
    ch: CHParams,
    mass_kg: float,
    r_max_hat: float = 500.0,
    n_points: int = 4000,
    hydrostatic_only: bool = True,
) -> GravityGPEResult:
    """
    Run Φ extraction on the Schwarzschild depletion ansatz (sanity check).

    Default hydrostatic_only=True: exterior Q=0 for (1−r_s/r)² tail; Newtonian
    matching uses the pressure-gradient channel (framework §5.2).
    """
    r_hat = np.linspace(0.5, r_max_hat, n_points)
    r_m = r_hat * ch.xi
    rho_norm = schwarzschild_depletion_profile(ch, mass_kg, r_m)
    if hydrostatic_only:
        phi = hydrostatic_only_phi(ch, r_m, rho_norm)
    else:
        phi = effective_phi(ch, r_m, rho_norm)
    accel = -np.gradient(phi, r_m)
    g_eff = np.abs(accel) * r_m**2 / mass_kg
    r_s = G_MEAS * mass_kg / C**2
    # Fit where linear depletion ansatz holds: r >> r_core and r >> r_s
    min_fit_r = max(10.0 * ch.xi, 8.0 * r_s)
    slope, intercept, r_lo, r_hi, g_mean = extract_newton_fit(
        r_m, accel, mass_kg, fit_fraction_lo=0.08, fit_fraction_hi=0.55, min_r_m=min_fit_r
    )
    newton_factor = slope
    alpha_g_far = g_mean * ch.rho_in / (ch.c_s**2 * ch.xi) if g_mean > 0 else 0.0
    psi = np.sqrt(np.clip(rho_norm, 0.0, None)).astype(complex)

    return GravityGPEResult(
        ch=ch,
        mass_kg=mass_kg,
        r_hat=r_hat,
        r_m=r_m,
        psi=psi,
        rho_norm=rho_norm,
        v0=0.0,
        rc_hat=0.0,
        mu=0.0,
        mass_deficit_kg=mass_kg,
        r_s_schwarzschild_m=r_s,
        phi_total=phi,
        accel_m_s2=accel,
        g_eff=g_eff,
        alpha_g_far=alpha_g_far,
        alpha_g_fit=slope,
        newton_slope=slope,
        newton_intercept=intercept,
        r_fit_lo_m=r_lo,
        r_fit_hi_m=r_hi,
        solver="analytic",
    )
