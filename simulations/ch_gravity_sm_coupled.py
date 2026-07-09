"""
Coupled ψ–Φ iteration — self-consistent S_M on inner ball + matched tail (Route G5c).

Outer Picard loop:
  1. GP with vacancy sink + Φ feedback on [0, r_join]
  2. Append Schwarzschild tail
  3. Poisson solve for Φ from (ρ − ρ_in)/ρ_in
  4. Repeat until ‖Φ − Φ_prev‖ and ‖ρ − ρ_prev‖ small

Compare to slaved v3 (Φ_hydro only) and report N = |a|/(GM/r²).

  python ch_gravity_sm_coupled.py
  python ch_gravity_sm_coupled.py --outer 12 --g0 0.5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams, C, G_MEAS
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    extend_schwarzschild_tail,
    extract_newton_fit,
    hydrostatic_coupling,
    hydro_newton_at_alpha_ref,
    mass_deficit_from_profile,
    mass_from_rs_hat,
    quantum_potential_radial,
    radial_laplacian,
    sm_sink_coefficient,
    solve_gravity_sm_v3,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha

OUTPUT = Path(__file__).parent / "output"


@dataclass
class CoupledSMResult:
    ch: CHParams
    r_hat: np.ndarray
    r_m: np.ndarray
    rho_norm: np.ndarray
    phi_j_kg: np.ndarray
    psi: np.ndarray
    mu: float
    g0: float
    sigma_hat: float
    rs_hat: float
    r_join_hat: float
    n_outer_iters: int
    phi_residual: float
    rho_residual: float
    newton_coupled: float
    newton_slaved_hydro: float
    newton_v3_ref: float
    z_phi: float
    coupling_eta: float
    coupling_coeff: float
    poisson_mode: str = "split"
    rho_core: float = float("nan")
    rho_core_v3: float = float("nan")
    rho_at_015: float = float("nan")
    m_inner_kg: float = float("nan")
    m_inner_v3_kg: float = float("nan")
    newton_coupled_q: float = float("nan")
    q_over_phi_core: float = float("nan")
    phi_g_j_kg: np.ndarray | None = None
    z_g_3rs: float = float("nan")
    z_g_20xi: float = float("nan")
    z_gr_3rs: float = float("nan")
    z_gr_20xi: float = float("nan")
    gp_mode: str = "flat"


def covariant_laplacian_radial(psi: np.ndarray, r: np.ndarray, g_rr: np.ndarray) -> np.ndarray:
    """
    Scalar Laplacian on static spatial metric ds² = g_rr dr² + r² dΩ².

    For ψ(r): ∇²_g ψ = r^{-2} ∂_r [r² g^{rr} ∂_r ψ] = r^{-2} ∂_r [(r²/g_rr) ∂_r ψ].
    """
    r = np.asarray(r, dtype=float)
    g_rr = np.clip(np.asarray(g_rr, dtype=float), 1e-12, None)
    psi = np.asarray(psi, dtype=complex)
    flux = (r**2 / g_rr) * np.gradient(psi, r)
    lap = np.gradient(flux, r) / np.clip(r**2, 1e-30, None)
    return lap


def z_g_from_identification(ch: CHParams, alpha_g: float) -> float:
    """
    Metric amplitude bridge Z_g: Φ_g = Φ_dyn / Z_g on the exterior tail.

    At grain α_G^hydro,ref: linear Φ_hydro has twice the chronos amplitude of
    Φ_g = −GM/(2r) that matches g_00 = −(1 − r_s/r) with repo r_s = GM/c².

    Z_g = 4 α_G c_s² / (m_grain c²) = 2 × hydrostatic_newton_factor(α_G).
    """
    from ch_gpe_gravity import hydrostatic_newton_factor

    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_g)
    return float(2.0 * hydrostatic_newton_factor(ch_ref))


def phi_metric_exterior(
    r_m: np.ndarray,
    phi_dyn: np.ndarray,
    mass_kg: float,
    r_join_m: float,
) -> np.ndarray:
    """Φ_g for metric sector: inner from dynamics; exterior −GM/(2r)."""
    phi_g = np.asarray(phi_dyn, dtype=float).copy()
    mask = r_m >= r_join_m - 1e-12
    phi_g[mask] = -0.5 * G_MEAS * mass_kg / np.clip(r_m[mask], 1e-30, None)
    return phi_g


def phi_g_from_dyn(
    phi_dyn: np.ndarray,
    r_m: np.ndarray,
    r_join_m: float,
    *,
    mass_kg: float | None = None,
    z_g: float | None = None,
) -> np.ndarray:
    """
    Build Φ_g from Φ_dyn.

    dual: metric overlay −GM/(2r) on exterior (needs mass_kg).
    bridge: Φ_g = Φ_dyn / Z_g on exterior (needs z_g).
    """
    if mass_kg is not None:
        return phi_metric_exterior(r_m, phi_dyn, mass_kg, r_join_m)
    if z_g is not None:
        phi_g = np.asarray(phi_dyn, dtype=float).copy()
        mask = r_m >= r_join_m - 1e-12
        phi_g[mask] = phi_dyn[mask] / z_g
        return phi_g
    raise ValueError("phi_g_from_dyn needs mass_kg (dual) or z_g (bridge)")


def z_clock_from_phi(phi_emit: float) -> float:
    """z = 1/√(1+2Φ/c²) − 1 with Φ(∞)=0."""
    omega = float(np.sqrt(max(1.0 + 2.0 * phi_emit / C**2, 1e-30)))
    return float(1.0 / omega - 1.0)


def z_clock_gr(r_emit_m: float, r_s_m: float) -> float:
    return float(1.0 / np.sqrt(1.0 - r_s_m / max(r_emit_m, r_s_m * 1.001)) - 1.0)


def z_phi_from_identification(ch: CHParams, alpha_g: float, lam_g: float) -> float:
    """
    Z_Φ so Poisson source (λ_g/Z_Φ)(ρ_norm−1) matches hydrostatic scale α_G(c_s²/m)(ρ_norm−1).

    EL: ∇²Φ = (λ_g/Z_Φ)(ρ_norm − 1); slaved Φ_hydro = α_G(c_s²/m)(ρ_norm − 1).
    """
    hydro_scale = alpha_g * ch.c_s**2 / ch.m_grain
    return lam_g / max(hydro_scale, 1e-99)


def poisson_phi_radial_dirichlet(
    r_m: np.ndarray,
    source: np.ndarray,
    phi_join: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve ∇²Φ = S on [0, r_join] with Φ(r_join) = phi_join, regular at origin.

    Integrate outward from r=0, then fix offset to match join value.
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
    phi[0] = 0.0
    for i in range(n - 1):
        phi[i + 1] = phi[i] + dphi_dr[i] * dr
    phi += phi_join - phi[-1]
    dphi_dr = np.gradient(phi, r)
    return phi, dphi_dr


def poisson_phi_radial_full(
    r_m: np.ndarray,
    source: np.ndarray,
    phi_far: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve ∇²Φ = S on [0, r_max] with Φ(r_max) = phi_far, regular at origin.

    Default phi_far=0; use phi_far=Φ_hydro(r_max) to anchor exterior amplitude.
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
    phi += phi_far - phi[-1]
    dphi_dr = np.gradient(phi, r)
    return phi, dphi_dr


def solve_phi_profile(
    ch: CHParams,
    r_hat: np.ndarray,
    r_in: np.ndarray,
    psi_in: np.ndarray,
    rho_norm: np.ndarray,
    rs_hat: float,
    r_join_hat: float,
    alpha_ref: float,
    lam_g: float,
    z_phi: float,
    *,
    poisson_mode: str = "split",
) -> np.ndarray:
    """Poisson for Φ: split (inner Dirichlet + hydro tail) or full grid."""
    r_m = r_hat * ch.xi
    rho_join = max((1.0 - rs_hat / r_join_hat) ** 2, 1e-6)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)

    if poisson_mode in ("full", "dual", "bridge"):
        # dual/bridge: iteration uses split/hydro Φ_dyn; Φ_g built post-loop (G9b).
        if poisson_mode == "full":
            return hydrostatic_coupling(ch_ref, rho_norm)

    if poisson_mode == "full_el":
        source = (lam_g / z_phi) * (rho_norm - 1.0)
        phi_new, _ = poisson_phi_radial_fd(r_m, source, phi_far=0.0)
        return phi_new

    r_in_m = r_in * ch.xi
    source_inner = (lam_g / z_phi) * (np.abs(psi_in) ** 2 - 1.0)

    if poisson_mode == "metric":
        # G9: metric-consistent exterior Φ = −GM/(2r) (repo r_s = GM/c²).
        mass_kg = mass_from_rs_hat(ch, rs_hat)
        r_join_m = r_join_hat * ch.xi
        phi_join = float(-0.5 * G_MEAS * mass_kg / r_join_m)
        phi_inner_new, _ = poisson_phi_radial_dirichlet(r_in_m, source_inner, phi_join)
        phi_new = np.zeros(len(r_m), dtype=float)
        n_inner = len(r_in)
        phi_new[:n_inner] = np.interp(r_hat[:n_inner], r_in, phi_inner_new)
        mask_ext = r_m >= r_join_m - 1e-12
        phi_new[mask_ext] = -0.5 * G_MEAS * mass_kg / np.clip(r_m[mask_ext], 1e-30, None)
        return phi_new

    phi_join = float(alpha_ref * ch.c_s**2 / ch.m_grain * (rho_join - 1.0))
    phi_inner_new, _ = poisson_phi_radial_dirichlet(r_in_m, source_inner, phi_join)
    phi_tail = hydrostatic_coupling(ch_ref, rho_norm)
    phi_new = phi_tail.copy()
    n_inner = len(r_in)
    phi_new[:n_inner] = np.interp(r_hat[:n_inner], r_in, phi_inner_new)
    return phi_new


def rho_at_hat(r_hat: np.ndarray, rho_norm: np.ndarray, target: float) -> float:
    """Interpolate ρ at dimensionless radius target."""
    return float(np.interp(target, r_hat, rho_norm))


def rho_core_value(rho_norm: np.ndarray, *, n_frac: float = 0.08) -> float:
    """Minimum ρ in inner core (avoid r=0 boundary artefact)."""
    n = max(4, int(len(rho_norm) * n_frac))
    return float(np.min(rho_norm[:n]))


def inner_mass_deficit(
    ch: CHParams,
    r_hat: np.ndarray,
    rho_norm: np.ndarray,
    r_join_hat: float,
) -> float:
    """Mass deficit integrated on inner ball r_hat <= r_join."""
    mask = r_hat <= r_join_hat + 1e-9
    return mass_deficit_from_profile(ch, r_hat[mask] * ch.xi, rho_norm[mask])


def poisson_phi_radial_fd(
    r_m: np.ndarray,
    source: np.ndarray,
    phi_far: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference ∇²Φ = S on uniform radial grid.

    BC: regular at origin, Φ(r_max) = phi_far (default 0 = infinity).
    """
    r = np.asarray(r_m, dtype=float)
    s = np.asarray(source, dtype=float)
    n = len(r)
    dr = r[1] - r[0]
    if n < 4:
        raise ValueError("grid too small for FD Poisson")

    a = np.zeros(n - 2)
    b = np.zeros(n - 2)
    c = np.zeros(n - 2)
    rhs = np.zeros(n - 2)

    # i = 0: 3(Φ_1 - Φ_0)/dr² = S_0
    b[0] = 3.0 / dr**2
    c[0] = -3.0 / dr**2
    rhs[0] = s[0]

    for i in range(1, n - 2):
        ri = r[i]
        rip = r[i + 1]
        rim = r[i - 1]
        a[i] = -rim**2 / dr**2
        b[i] = (rip**2 + rim**2) / dr**2
        c[i] = -rip**2 / dr**2
        rhs[i] = ri**2 * s[i]

    # Thomas algorithm (interior 0..n-3); Φ_{n-1} = phi_far
    rip_last = r[-1]
    rim_last = r[-2]
    a_last = -rim_last**2 / dr**2
    b_last = (rip_last**2 + rim_last**2) / dr**2
    rhs_last = rip_last**2 * s[-2] + (rip_last**2 / dr**2) * phi_far

    # extend system for i = n-2
    a_ext = np.append(a, a_last)
    b_ext = np.append(b, b_last)
    c_ext = np.append(c, 0.0)
    rhs_ext = np.append(rhs, rhs_last)

    phi_int = _thomas_solve(a_ext, b_ext, c_ext, rhs_ext)
    phi = np.zeros(n)
    phi[0] = phi_int[0]
    phi[1:-1] = phi_int[1:]
    phi[-1] = phi_far
    dphi_dr = np.gradient(phi, r)
    return phi, dphi_dr


def _thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Tridiagonal solve (a lower, b diag, c upper)."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def poisson_phi_radial_full(
    r_m: np.ndarray,
    source: np.ndarray,
    phi_far: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy flux integration Poisson (kept for cross-checks).

    Prefer poisson_phi_radial_fd for full-grid solves.
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
    phi += phi_far - phi[-1]
    dphi_dr = np.gradient(phi, r)
    return phi, dphi_dr


def coupling_coeff_el(ch: CHParams, lam_g: float) -> float:
    """SI coefficient in μψ += (λ_g Φ)/(c² ρ_in) ψ from Eq. (app-el-psi)."""
    return lam_g / (ch.rho_in * C**2)


def solve_gp_inner_with_phi(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    r_join_hat: float,
    rho_join: float,
    phi_j_kg: np.ndarray,
    coupling_coeff: float,
    *,
    n_points: int = 1000,
    n_iter: int = 8000,
    dt: float = 0.002,
    artificial_coupling: bool = False,
    gp_mode: str = "flat",
    phi_g_for_grr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inner GP with sink + Φ coupling (dimensionless φ_hat = Φ m_grain/c_s²)."""
    r = np.linspace(0.0, r_join_hat, n_points)
    r[0] = max(r[1] * 0.15, 1e-6)
    dr = r[1] - r[0]
    gamma = sm_sink_coefficient(r, g0, sigma_hat)
    phi = np.asarray(phi_j_kg, dtype=float)
    if len(phi) != len(r):
        phi = np.interp(r, np.linspace(0, r_join_hat, len(phi)), phi)

    grr = None
    if gp_mode == "covariant":
        phi_grr = np.asarray(phi_g_for_grr if phi_g_for_grr is not None else phi, dtype=float)
        if len(phi_grr) != len(r):
            phi_grr = np.interp(r, np.linspace(0, r_join_hat, len(phi_grr)), phi_grr)
        grr = np.clip(1.0 + 2.0 * phi_grr / C**2, 0.25, 4.0)

    psi_join = np.sqrt(max(rho_join, 1e-12))
    psi = np.full(n_points, np.sqrt(max(0.5 * (rho_join + 1.0), 1e-6)), dtype=complex)
    psi[-1] = psi_join + 0j
    mu = 0.0

    for _ in range(n_iter):
        rho = np.clip(np.abs(psi) ** 2, 1e-12, 1.0)
        psi = np.sqrt(rho).astype(complex)
        if grr is None:
            lap = np.zeros(n_points, dtype=complex)
            lap[0] = (psi[1] - psi[0]) / dr**2 + 2.0 * (psi[1] - psi[0]) / (r[0] * dr)
            lap[1:-1] = (
                (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dr**2
                + 2.0 * (psi[1:-1] - psi[:-2]) / (r[1:-1] * dr)
            )
            lap[-1] = (psi[-1] - 2.0 * psi[-2]) / dr**2 + 2.0 * (psi[-1] - psi[-2]) / (r[-1] * dr)
        else:
            lap = covariant_laplacian_radial(psi, r, grr).astype(complex)
        hpsi = -0.5 * lap + (1.0 - rho) * psi - gamma * psi
        if artificial_coupling:
            phi_scale = max(float(np.max(np.abs(phi))), 1e-99)
            hpsi += coupling_coeff * (phi / phi_scale) * gamma * psi
        else:
            hpsi += coupling_coeff * phi * psi
        mu_new = float(np.real(np.vdot(psi, hpsi) / np.vdot(psi, psi)))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        rho_new = np.clip(np.abs(psi_new) ** 2, 1e-12, 1.0)
        psi_new = np.sqrt(rho_new).astype(complex)
        psi_new[-1] = psi_join + 0j
        psi_new[0] = psi_new[1]
        if abs(mu_new - mu) < 1e-9 and float(np.max(np.abs(psi_new - psi))) < 1e-8:
            psi, mu = psi_new, mu_new
            break
        psi, mu = psi_new, mu_new

    return r, psi, mu


def newton_factor_from_phi(
    ch: CHParams,
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    phi_j_kg: np.ndarray,
    mass_kg: float,
    r_join_hat: float,
    *,
    include_q: bool = False,
) -> float:
    """N = |a|/(GM/r²) from Φ_total = Q + Φ_coupled on exterior."""
    join_m = r_join_hat * ch.xi
    mask = r_m >= join_m
    if int(np.sum(mask)) < 8:
        return float("nan")
    r_fit = r_m[mask]
    rho_fit = rho_norm[mask]
    phi_fit = phi_j_kg[mask]
    phi_q = quantum_potential_radial(ch, r_fit, rho_fit) if include_q else 0.0
    phi_tot = phi_q + phi_fit
    accel = -np.gradient(phi_tot, r_fit)
    slope, _, _, _, _ = extract_newton_fit(
        r_fit,
        accel,
        mass_kg,
        fit_fraction_lo=0.12,
        fit_fraction_hi=0.5,
        min_r_m=join_m * 1.02,
    )
    return float(slope)


def solve_coupled_sm(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
    r_join_hat: float = 12.0,
    r_max_hat: float = 300.0,
    *,
    n_outer: int = 15,
    omega: float = 0.35,
    coupling_scale: float = 0.15,
    poisson_mode: str = "split",
    artificial_coupling: bool = False,
    gp_mode: str = "flat",
) -> CoupledSMResult:
    """Picard ψ–Φ iteration with matched Schwarzschild tail."""
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    z_phi = z_phi_from_identification(ch, alpha_ref, lam_g)
    coeff = coupling_coeff_el(ch, lam_g)

    rho_join = max((1.0 - rs_hat / r_join_hat) ** 2, 1e-6)
    mass_kg = mass_from_rs_hat(ch, rs_hat)

    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    v3_init = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)
    r_phi_init = np.linspace(0.0, r_join_hat, 1000)
    mask_init = v3_init.r_hat <= r_join_hat + 1e-9
    phi_inner = np.interp(
        r_phi_init,
        v3_init.r_hat[mask_init],
        hydrostatic_coupling(ch_ref, v3_init.rho_norm[mask_init]),
    )
    rho_prev = None
    phi_prev = None
    phi_g_grr = None
    n_done = 0

    for outer in range(n_outer):
        gp_boost = coupling_scale if artificial_coupling else coeff * coupling_scale
        r_in, psi_in, mu = solve_gp_inner_with_phi(
            ch,
            g0,
            sigma_hat,
            r_join_hat,
            rho_join,
            phi_inner,
            gp_boost,
            n_iter=8000,
            artificial_coupling=artificial_coupling,
            gp_mode=gp_mode,
            phi_g_for_grr=phi_g_grr,
        )
        r_hat, psi_full = extend_schwarzschild_tail(
            r_in, psi_in, rs_hat, r_max_hat
        )
        r_m = r_hat * ch.xi
        rho_norm = np.clip(np.abs(psi_full) ** 2, 0.0, 1.0)

        r_in_m = r_in * ch.xi
        source_inner = (lam_g / z_phi) * (np.abs(psi_in) ** 2 - 1.0)
        phi_join = float(
            alpha_ref * ch.c_s**2 / ch.m_grain * (rho_join - 1.0)
        )
        phi_new = solve_phi_profile(
            ch,
            r_hat,
            r_in,
            psi_in,
            rho_norm,
            rs_hat,
            r_join_hat,
            alpha_ref,
            lam_g,
            z_phi,
            poisson_mode=poisson_mode,
        )

        if phi_prev is not None:
            phi_blend = (1.0 - omega) * phi_prev + omega * phi_new
            n_inner = len(r_in)
            phi_inner_blend = np.interp(
                np.linspace(0, r_join_hat, len(phi_inner)),
                r_hat[:n_inner],
                phi_blend[:n_inner],
            )
        else:
            phi_blend = phi_new
            n_inner = len(r_in)
            phi_inner_blend = np.interp(
                np.linspace(0, r_join_hat, len(phi_inner)),
                r_hat[:n_inner],
                phi_new[:n_inner],
            )

        phi_res = float(
            np.max(np.abs(phi_blend - phi_prev)) / max(np.max(np.abs(phi_prev)), 1e-99)
        ) if phi_prev is not None else 1.0
        rho_res = float(
            np.max(np.abs(rho_norm - rho_prev)) / max(np.max(rho_prev), 1e-99)
        ) if rho_prev is not None else 1.0

        if gp_mode == "covariant" and poisson_mode in ("dual", "bridge"):
            r_join_m = r_join_hat * ch.xi
            z_g_loop = z_g_from_identification(ch, alpha_ref)
            if poisson_mode == "dual":
                phi_g_loop = phi_metric_exterior(r_m, phi_blend, mass_kg, r_join_m)
            else:
                phi_g_loop = phi_g_from_dyn(phi_blend, r_m, r_join_m, z_g=z_g_loop)
            phi_g_grr = np.interp(r_in, r_hat[:n_inner], phi_g_loop[:n_inner])

        phi_prev = phi_blend
        phi_inner = phi_inner_blend
        rho_prev = rho_norm.copy()
        n_done = outer + 1

        if outer > 0 and phi_res < 0.02 and rho_res < 0.02:
            break

    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    phi_hydro = hydrostatic_coupling(ch_ref, rho_norm)
    n_coupled = newton_factor_from_phi(
        ch, r_m, rho_norm, phi_blend, mass_kg, r_join_hat
    )
    n_coupled_q = newton_factor_from_phi(
        ch, r_m, rho_norm, phi_blend, mass_kg, r_join_hat, include_q=True
    )
    n_slaved = newton_factor_from_phi(
        ch, r_m, rho_norm, phi_hydro, mass_kg, r_join_hat
    )

    v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)
    n_v3 = hydro_newton_at_alpha_ref(v3)
    rho_core = rho_core_value(rho_norm)
    rho_core_v3 = rho_core_value(v3.rho_norm)
    rho_015 = rho_at_hat(r_hat, rho_norm, 0.15)
    m_inner = inner_mass_deficit(ch, r_hat, rho_norm, r_join_hat)
    m_inner_v3 = inner_mass_deficit(ch, v3.r_hat, v3.rho_norm, r_join_hat)
    join_mask = (r_hat >= 0.85 * r_join_hat) & (r_hat <= r_join_hat)
    if int(np.sum(join_mask)) >= 4:
        r_q = r_m[join_mask]
        rho_q = rho_norm[join_mask]
        phi_q_region = phi_blend[join_mask]
        q_region = quantum_potential_radial(ch, r_q, rho_q)
        q_over_phi = float(
            np.max(np.abs(q_region)) / max(np.max(np.abs(phi_q_region)), 1e-99)
        )
    else:
        q_over_phi = float("nan")

    phi_g_arr = None
    z_g_3rs = float("nan")
    z_g_20xi = float("nan")
    z_gr_3rs = float("nan")
    z_gr_20xi = float("nan")
    if poisson_mode in ("dual", "bridge"):
        r_join_m = r_join_hat * ch.xi
        alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
        z_g = z_g_from_identification(ch, alpha_ref)
        if poisson_mode == "dual":
            phi_g_arr = phi_metric_exterior(r_m, phi_blend, mass_kg, r_join_m)
        else:
            phi_g_arr = phi_g_from_dyn(
                phi_blend, r_m, r_join_m, z_g=z_g
            )
        r_emit_3 = 3.0 * rs_hat * ch.xi
        r_emit_20 = 20.0 * ch.xi
        i3 = int(np.argmin(np.abs(r_m - r_emit_3)))
        i20 = int(np.argmin(np.abs(r_m - r_emit_20)))
        z_g_3rs = z_clock_from_phi(float(phi_g_arr[i3]))
        z_g_20xi = z_clock_from_phi(float(phi_g_arr[i20]))
        z_gr_3rs = z_clock_gr(r_emit_3, rs_hat * ch.xi)
        z_gr_20xi = z_clock_gr(r_emit_20, rs_hat * ch.xi)

    return CoupledSMResult(
        ch=ch,
        r_hat=r_hat,
        r_m=r_m,
        rho_norm=rho_norm,
        phi_j_kg=phi_blend,
        psi=psi_full,
        mu=mu,
        g0=g0,
        sigma_hat=sigma_hat,
        rs_hat=rs_hat,
        r_join_hat=r_join_hat,
        n_outer_iters=n_done,
        phi_residual=phi_res,
        rho_residual=rho_res,
        newton_coupled=n_coupled,
        newton_slaved_hydro=n_slaved,
        newton_v3_ref=n_v3,
        z_phi=z_phi,
        coupling_eta=coeff * coupling_scale,
        coupling_coeff=coeff,
        poisson_mode=poisson_mode,
        rho_core=rho_core,
        rho_core_v3=rho_core_v3,
        rho_at_015=rho_015,
        m_inner_kg=m_inner,
        m_inner_v3_kg=m_inner_v3,
        newton_coupled_q=n_coupled_q,
        q_over_phi_core=q_over_phi,
        phi_g_j_kg=phi_g_arr,
        z_g_3rs=z_g_3rs,
        z_g_20xi=z_g_20xi,
        z_gr_3rs=z_gr_3rs,
        z_gr_20xi=z_gr_20xi,
        gp_mode=gp_mode,
    )


def format_report(res: CoupledSMResult) -> str:
    lines = [
        "CH coupled ψ–Φ iteration (Route G5c)",
        f"xi = {res.ch.xi:.3e} m, r_s/xi = {res.rs_hat}, r_join/xi = {res.r_join_hat}",
        f"gamma0 = {res.g0:.4g}, sigma = {res.sigma_hat}",
        f"outer iterations = {res.n_outer_iters}, phi_res = {res.phi_residual:.3e}, rho_res = {res.rho_residual:.3e}",
        f"Z_Phi = {res.z_phi:.4e}, GP coupling (lambda_g Phi)/(c^2 rho_in) = {res.coupling_coeff:.4e}",
        f"  effective scale in loop = {res.coupling_eta:.4e} (x coupling_scale)",
        f"Poisson mode = {res.poisson_mode}, GP mode = {res.gp_mode}",
        f"rho_core = {res.rho_core:.6f} (v3 ref {res.rho_core_v3:.6f})",
        f"M_inner = {res.m_inner_kg:.4e} kg (v3 ref {res.m_inner_v3_kg:.4e} kg)",
        f"|Q|/|Phi| near join ≈ {res.q_over_phi_core:.3e}",
        "",
        "Newton factor N = |a|/(GM/r^2) on exterior:",
        f"  coupled Phi (Poisson+GP loop) = {res.newton_coupled:.4f}",
        f"  coupled Phi + Q (Madelung)   = {res.newton_coupled_q:.4f}",
        f"  slaved Φ_hydro only         = {res.newton_slaved_hydro:.4f}",
        f"  v3 reference (laptop)     = {res.newton_v3_ref:.4f}",
    ]
    if res.poisson_mode in ("dual", "bridge") and res.phi_g_j_kg is not None:
        alpha_ref = alpha_g_required_for_hydrostatic_newton(res.ch)
        zg = z_g_from_identification(res.ch, alpha_ref)
        label = "metric overlay" if res.poisson_mode == "dual" else f"Φ_dyn/Z_g, Z_g={zg:.2f}"
        lines += [
            "",
            f"Metric sector (G9b {res.poisson_mode}, Φ_g: {label}):",
            f"  Z_g (identified) = {zg:.4f}",
            f"  z_g(3r_s)/z_GR = {res.z_g_3rs / max(res.z_gr_3rs, 1e-30):.3f}",
            f"  z_g(20ξ)/z_GR  = {res.z_g_20xi / max(res.z_gr_20xi, 1e-30):.3f}",
        ]
    lines += [
        "",
        "Reading:",
        "  Self-consistent loop: GP feels Φ; Φ sourced by ρ deficit (Poisson).",
        "  Slaved hydrostatic Φ_hydro is the fixed point target at grain α_G.",
        "  Convergence → coupled N should approach slaved/v3 (~0.98).",
        "  At grain alpha_G, Phi back-reaction on rho is weak; loop finds hydrostatic fixed point.",
    ]
    if res.poisson_mode == "dual":
        lines.append("  dual mode: Φ_dyn drives loop; Φ_g = −GM/(2r) exterior for g_00 clocks.")
    elif res.poisson_mode == "bridge":
        lines.append("  bridge mode: Φ_dyn drives loop; Φ_g = Φ_dyn/Z_g exterior (action ID).")
    return "\n".join(lines)


def plot_result(res: CoupledSMResult, out_png: Path) -> None:
    ch_ref = CHParams(xi=res.ch.xi, alpha_g=alpha_g_required_for_hydrostatic_newton(res.ch))
    phi_h = hydrostatic_coupling(ch_ref, res.rho_norm)
    mask = res.r_hat <= res.r_join_hat * 1.5

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(res.r_hat[mask], res.rho_norm[mask], "k-", label=r"$\rho/\rho_{\mathrm{in}}$")
    axes[0].set_xlabel(r"$\hat r$")
    axes[0].set_title("Density")
    axes[0].grid(alpha=0.3)

    axes[1].plot(res.r_hat, phi_h, "b--", label=r"$\Phi_{\mathrm{hydro}}$")
    axes[1].plot(res.r_hat, res.phi_j_kg, "r-", label=r"$\Phi_{\mathrm{dyn}}$")
    if res.phi_g_j_kg is not None:
        axes[1].plot(res.r_hat, res.phi_g_j_kg, "g-.", label=r"$\Phi_g$", alpha=0.85)
    axes[1].set_xlabel(r"$\hat r$")
    axes[1].set_ylabel(r"$\Phi$ [J/kg]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Potential")
    axes[1].grid(alpha=0.3)

    axes[2].bar(
        ["coupled", "slaved", "v3"],
        [res.newton_coupled, res.newton_slaved_hydro, res.newton_v3_ref],
        color=["C3", "C0", "C2"],
    )
    axes[2].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[2].set_ylabel(r"$N_{\mathrm{hydro}}$")
    axes[2].set_title("Exterior Newton")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--rs-hat", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--g0", type=float, default=None)
    parser.add_argument("--outer", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.35)
    parser.add_argument("--coupling-scale", type=float, default=0.15)
    parser.add_argument(
        "--poisson-mode",
        choices=("split", "metric", "dual", "bridge", "full", "full_el"),
        default="split",
        help="split: hydro tail; metric: −GM/(2r) tail; dual: split + Φ_g overlay; bridge: split + Φ_g=Φ_dyn/Z_g",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0 = args.g0 if args.g0 is not None else calibrate_g0_matched(args.rs_hat, sigma_hat=args.sigma)[0]

    res = solve_coupled_sm(
        ch,
        g0,
        args.sigma,
        args.rs_hat,
        n_outer=args.outer,
        omega=args.omega,
        coupling_scale=args.coupling_scale,
        poisson_mode=args.poisson_mode,
    )
    report = format_report(res)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ch_gravity_sm_coupled.txt").write_text(report + "\n")
    if not args.no_plot:
        plot_result(res, OUTPUT / "ch_gravity_sm_coupled.png")
        print(f"Wrote {OUTPUT}/ch_gravity_sm_coupled.png")
    print(f"Wrote {OUTPUT}/ch_gravity_sm_coupled.txt")


if __name__ == "__main__":
    main()
