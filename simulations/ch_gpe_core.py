"""
1D stationary Gross–Pitaevskii solver for CH Casimir-gap boundary problems.

The GPE step answers: given plate separation d and healing length ξ, what is
|∇ρ| at the measurement point (and near walls) from boundary conditions ψ=0
on metal surfaces?

Dimensionless form (lengths in units of ξ, ρ in units of ρ_in):

  μ ψ = (-½ ∂²/∂ẑ² + 1 - |ψ|²) ψ,   ẑ ∈ [0, d̂],   ψ(0)=ψ(d̂)=0

Physical density: ρ(z) = ρ_in |ψ(z)|²
Gradient ratio:   |∇ρ| / |∇ρ|_c = |∂|ψ|²/∂ẑ|   (since |∇ρ|_c = ρ_in/ξ)

See docs/ch-gradient-threshold-experiment.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ch_dispersion_core import CHParams, chi


@dataclass
class GPE1DResult:
    """Ground-state solution for ψ in a 1D box of width d_hat (in units of ξ)."""

    z_hat: np.ndarray
    psi: np.ndarray
    d_hat: float
    d_m: float
    xi_m: float
    rho_in: float
    grad_rho_crit: float
    mu: float

    @property
    def rho_norm(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    @property
    def grad_rho_norm(self) -> np.ndarray:
        """|∂|ψ|²/∂ẑ| — equals |∇ρ|/|∇ρ|_c in CH units."""
        return np.abs(np.gradient(self.rho_norm, self.z_hat))

    @property
    def grad_rho_si(self) -> np.ndarray:
        return self.grad_rho_norm * self.grad_rho_crit

    @property
    def grad_ratio_max(self) -> float:
        if self.d_hat > 200:
            return fine_grad_ratios(self.d_hat)[2]
        return float(np.max(self.grad_rho_norm))

    @property
    def chi_wall(self) -> float:
        g = self.grad_ratio_wall * self.grad_rho_crit
        return float(chi(g, self.grad_rho_crit))

    @property
    def grad_ratio_wall(self) -> float:
        """Peak |∂ρ/∂ẑ| in the wall boundary layer (ξ-resolved TF grid)."""
        return fine_grad_ratios(self.d_hat)[0]

    @property
    def grad_ratio_mid(self) -> float:
        """Bulk |∂ρ/∂ẑ| (central-half maximum on ξ-resolved TF grid)."""
        return fine_grad_ratios(self.d_hat)[1]

    @property
    def chi_mid(self) -> float:
        g = self.grad_ratio_mid * self.grad_rho_crit
        return float(chi(g, self.grad_rho_crit))

    @property
    def chi_max(self) -> float:
        g = self.grad_ratio_max * self.grad_rho_crit
        return float(chi(g, self.grad_rho_crit))


def thomas_fermi_grad_norm(z_hat: np.ndarray, d_hat: float) -> np.ndarray:
    """|∂ρ/∂ẑ| for ρ = tanh(ẑ) tanh(d̂ − ẑ)."""
    z = np.asarray(z_hat, dtype=float)
    t1 = np.tanh(np.clip(z, 0.0, None))
    t2 = np.tanh(np.clip(d_hat - z, 0.0, None))
    dt1 = 1.0 / np.cosh(np.clip(z, -20.0, 20.0)) ** 2
    dt2 = -1.0 / np.cosh(np.clip(d_hat - z, -20.0, 20.0)) ** 2
    return np.abs(dt1 * t2 + t1 * dt2)


def fine_grad_ratios(d_hat: float) -> tuple[float, float, float]:
    """
    |∇ρ|/|∇ρ|_c at wall and bulk on a ξ-resolved Thomas--Fermi grid.

    Wall: peak in the first ~1/8 of the box (boundary layer).
    Bulk (``grad_mid``): maximum in the central half --- not the geometric
    midpoint alone, which sits at a symmetry zero of tanh(z)tanh(d−z) when
    layers overlap.

    Returns (grad_wall, grad_mid, grad_max).
    """
    if d_hat <= 0.04:
        return 0.0, 0.0, 0.0
    z = np.linspace(0.02, max(d_hat - 0.02, 0.04), 4000)
    grad = thomas_fermi_grad_norm(z, d_hat)
    n = len(grad)
    q = max(n // 8, 2)
    grad_wall = float(np.max(grad[:q]))

    if d_hat > 20:
        grad_mid = 0.0
    else:
        i0, i1 = n // 4, 3 * n // 4
        grad_mid = float(np.max(grad[i0:i1])) if i1 > i0 else 0.0

    grad_max = float(np.max(grad))
    return grad_wall, grad_mid, grad_max


def thomas_fermi_box_profile(z_hat: np.ndarray, d_hat: float) -> np.ndarray:
    """
    Analytic 1D box ground-state approximation for d̂ ≫ 1:

      ρ(ẑ) ≈ tanh(ẑ) tanh(d̂ − ẑ)

    Valid when boundary layers (width ~ 1 in ξ units) do not overlap.
    """
    z = np.asarray(z_hat, dtype=float)
    rho = np.tanh(np.clip(z, 0.0, None)) * np.tanh(np.clip(d_hat - z, 0.0, None))
    rho[0] = rho[-1] = 0.0
    return rho


def solve_gpe_1d_box_analytic(d_hat: float, n_points: int = 2048) -> tuple[np.ndarray, np.ndarray, float]:
    """Thomas–Fermi analytic profile for wide gaps (d̂ large)."""
    z = np.linspace(0.0, d_hat, n_points)
    rho = thomas_fermi_box_profile(z, d_hat)
    psi = np.sqrt(np.clip(rho, 0.0, None)).astype(complex)
    return z, psi, 0.0


def solve_gpe_1d_box(
    d_hat: float,
    n_points: int = 512,
    n_iter: int = 8000,
    dt: float = 0.005,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Imaginary-time propagation for the 1D GP ground state in a hard-wall box.

    Dimensionless GP: μ ψ = (-½ ∂²/∂ẑ² + 1 - |ψ|²) ψ with ψ(0)=ψ(d̂)=0.
    Bulk far from walls → |ψ|² → 1. Lengths in units of healing length ξ.

    Returns (z_hat, psi, mu) with ψ(0)=ψ(d_hat)=0.
    """
    if d_hat <= 0:
        raise ValueError("d_hat must be positive")
    if n_points < 8:
        raise ValueError("n_points must be >= 8")

    z = np.linspace(0.0, d_hat, n_points)
    dz = z[1] - z[0]

    # Thomas–Fermi wall profile: ρ ≈ tanh(z)tanh(d−z) for d̂ ≫ 1
    rho0 = np.tanh(np.clip(z, 1e-12, None)) * np.tanh(np.clip(d_hat - z, 1e-12, None))
    psi = np.sqrt(np.clip(rho0, 0.0, None)).astype(complex)
    psi[0] = psi[-1] = 0.0

    def normalize_bulk(psi_arr: np.ndarray) -> np.ndarray:
        """Scale so central third has mean |ψ|² ≈ 1 (bulk ρ_in)."""
        n = len(psi_arr)
        i0, i1 = n // 4, 3 * n // 4
        if i1 <= i0:
            i0, i1 = 1, n - 1
        rho = np.abs(psi_arr) ** 2
        mid = float(np.mean(rho[i0:i1]))
        if mid > 1e-30:
            psi_arr = psi_arr / np.sqrt(mid)
        psi_arr[0] = psi_arr[-1] = 0.0
        return psi_arr

    psi = normalize_bulk(psi)
    mu = 0.0
    for _ in range(n_iter):
        rho = np.abs(psi) ** 2
        lap = np.zeros(n_points, dtype=complex)
        lap[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / dz**2
        hpsi = -0.5 * lap + (1.0 - rho) * psi
        mu_new = float(np.real(np.vdot(psi, hpsi) / np.vdot(psi, psi)))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        psi_new[0] = psi_new[-1] = 0.0
        psi_new = normalize_bulk(psi_new)

        dmu = abs(mu_new - mu)
        dpsi = float(np.max(np.abs(psi_new - psi)))
        psi = psi_new
        mu = mu_new
        if dmu < tol and dpsi < tol:
            break

    return z, psi, mu


def solve_casimir_gap(ch: CHParams, d_m: float, **solver_kw) -> GPE1DResult:
    """Solve 1D GP for parallel-plate gap separation d_m [m]."""
    d_hat = d_m / ch.xi
    n_points = int(solver_kw.pop("n_points", 512))
    analytic_threshold = float(solver_kw.pop("analytic_threshold", 200.0))

    if d_hat > analytic_threshold:
        # Numerical grid cannot resolve ξ-scale boundary layers when d̂ ≫ n_points
        z_hat, psi, mu = solve_gpe_1d_box_analytic(d_hat, n_points=max(n_points, 4096))
        method = "thomas_fermi_analytic"
    else:
        z_hat, psi, mu = solve_gpe_1d_box(d_hat, n_points=n_points, **solver_kw)
        method = "imaginary_time"

    result = GPE1DResult(
        z_hat=z_hat,
        psi=psi,
        d_hat=d_hat,
        d_m=d_m,
        xi_m=ch.xi,
        rho_in=ch.rho_in,
        grad_rho_crit=ch.grad_rho_crit,
        mu=mu,
    )
    result.method = method  # type: ignore[attr-defined]
    return result


def scan_gap_separations(
    ch: CHParams,
    d_m_array: np.ndarray,
    **solver_kw,
) -> list[GPE1DResult]:
    return [solve_casimir_gap(ch, float(d), **solver_kw) for d in d_m_array]


def tidal_grad_ratio(ch: CHParams, mass_kg: float, r_m: float) -> float:
    """Far-field tidal |∇ρ|/|∇ρ|_c (e.g. Earth surface)."""
    if r_m <= 0:
        return 0.0
    from ch_dispersion_core import C, G_MEAS

    grad_env = ch.rho_in * 2.0 * G_MEAS * mass_kg / (C**2 * r_m**2)
    return grad_env / ch.grad_rho_crit


def wall_scaling_ratio(ch: CHParams, d_m: float) -> float:
    """Doc scaling estimate |∇ρ|_wall/|∇ρ|_c ~ ξ/d (order-of-magnitude)."""
    return ch.xi / d_m


def predict_alpha_eff(chi_val: float, alpha_max: float = 0.12) -> float:
    """Casimir ripple amplitude from gradient gate value."""
    return alpha_max * chi_val


def predict_visibility_dip(chi_val: float, dip_max: float = 0.15) -> float:
    """Mach–Zehnder visibility dip from gradient gate value."""
    return dip_max * chi_val


# --- 2D sphere–plate (axisymmetric Thomas–Fermi ansatz) ---


def sphere_plate_gap_hat(r_hat: np.ndarray, d_min_hat: float, R_hat: float) -> np.ndarray:
    """Gap height ĥ(r̂) between flat plate and sphere (parabolic approximation, r̂ ≪ R̂)."""
    return d_min_hat + r_hat**2 / (2.0 * R_hat)


def rho_sphere_plate_2d(
    r_hat: np.ndarray, z_hat: np.ndarray, d_min_hat: float, R_hat: float
) -> np.ndarray:
    """
    Thomas–Fermi ρ(r̂,ẑ) in sphere–plate geometry (dimensionless |ψ|²).

    ρ = tanh(ẑ) tanh(ĥ(r̂)−ẑ) for 0 ≤ ẑ ≤ ĥ(r̂), else 0.
    """
    r = np.asarray(r_hat, dtype=float)
    z = np.asarray(z_hat, dtype=float)
    h = sphere_plate_gap_hat(r, d_min_hat, R_hat)
    rho = np.tanh(np.clip(z, 0.0, None)) * np.tanh(np.clip(h - z, 0.0, None))
    return np.where(z <= h, rho, 0.0)


def grad_sphere_plate_2d_analytic(
    r_hat: np.ndarray, z_hat: np.ndarray, d_min_hat: float, R_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ∂ρ/∂r̂ and ∂ρ/∂ẑ for the Thomas–Fermi sphere–plate profile."""
    r = np.asarray(r_hat, dtype=float)
    z = np.asarray(z_hat, dtype=float)
    h = sphere_plate_gap_hat(r, d_min_hat, R_hat)
    t_z = np.tanh(np.clip(z, 0.0, None))
    t_g = np.tanh(np.clip(h - z, 0.0, None))
    s_z = 1.0 / np.cosh(np.clip(z, -20.0, 20.0)) ** 2
    s_g = 1.0 / np.cosh(np.clip(h - z, -20.0, 20.0)) ** 2
    drho_dz = s_z * t_g - t_z * s_g
    drho_dr = t_z * s_g * (r / max(R_hat, 1e-30))
    return drho_dr, drho_dz


@dataclass
class GPE2DSpherePlateResult:
    """Gradient metrics for sphere–plate Casimir geometry."""

    d_min_m: float
    R_m: float
    xi_m: float
    d_min_hat: float
    R_hat: float
    rho_in: float
    grad_rho_crit: float
    grad_ratio_wall: float
    grad_ratio_mid: float
    grad_ratio_rim: float
    grad_ratio_radial: float
    chi_wall: float
    chi_mid: float
    chi_rim: float
    chi_radial: float

    @property
    def grad_ratio_max(self) -> float:
        return max(
            self.grad_ratio_wall,
            self.grad_ratio_mid,
            self.grad_ratio_rim,
            self.grad_ratio_radial,
        )


def solve_sphere_plate(
    ch: CHParams,
    d_min_m: float,
    R_m: float,
    rim_r_hat: float = 1.0,
) -> GPE2DSpherePlateResult:
    """
    2D axisymmetric Thomas–Fermi sphere–plate ground state.

    Probes:
      - wall: (r̂,ẑ) → (0, 0⁺)
      - mid:  (0, d̂_min/2) when d̂_min > 0.2
      - rim:  (rim_r_hat, 0⁺) — curvature-enhanced contact perimeter
    """
    if d_min_m <= 0 or R_m <= 0:
        raise ValueError("d_min and R must be positive")

    d_min_hat = d_min_m / ch.xi
    R_hat = R_m / ch.xi

    # Wall (r=0, z→0+): reduces to 1D parallel-plate wall
    _, drho_dz_wall = grad_sphere_plate_2d_analytic(
        np.array([0.0]), np.array([0.02]), d_min_hat, R_hat
    )
    grad_wall = float(np.abs(drho_dz_wall[0]))

    # Midpoint along axis
    if d_min_hat > 0.2:
        z_mid = d_min_hat / 2.0
        _, drho_dz_mid = grad_sphere_plate_2d_analytic(
            np.array([0.0]), np.array([z_mid]), d_min_hat, R_hat
        )
        grad_mid = float(np.abs(drho_dz_mid[0]))
    else:
        z_m = np.linspace(0.02, max(d_min_hat - 0.02, 0.05), 500)
        _, drho_dz = grad_sphere_plate_2d_analytic(np.zeros_like(z_m), z_m, d_min_hat, R_hat)
        grad_mid = float(np.max(np.abs(drho_dz)))

    # Radial curvature probe at rim (ẑ ~ 1 in healing-length units)
    r_rim = min(rim_r_hat, 0.2 * R_hat)
    z_probe = min(max(1.0, 0.1 * d_min_hat), d_min_hat * 0.45)
    drho_dr, drho_dz = grad_sphere_plate_2d_analytic(
        np.array([r_rim]), np.array([z_probe]), d_min_hat, R_hat
    )
    grad_rim = float(np.sqrt(drho_dr[0] ** 2 + drho_dz[0] ** 2))
    grad_radial = float(np.abs(drho_dr[0]))

    def _chi(ratio: float) -> float:
        return float(chi(ratio * ch.grad_rho_crit, ch.grad_rho_crit))

    return GPE2DSpherePlateResult(
        d_min_m=d_min_m,
        R_m=R_m,
        xi_m=ch.xi,
        d_min_hat=d_min_hat,
        R_hat=R_hat,
        rho_in=ch.rho_in,
        grad_rho_crit=ch.grad_rho_crit,
        grad_ratio_wall=grad_wall,
        grad_ratio_mid=grad_mid,
        grad_ratio_rim=grad_rim,
        grad_ratio_radial=grad_radial,
        chi_wall=_chi(grad_wall),
        chi_mid=_chi(grad_mid),
        chi_rim=_chi(grad_rim),
        chi_radial=_chi(grad_radial),
    )


def scan_sphere_plate_R(
    ch: CHParams,
    d_min_m: float,
    R_m_array: np.ndarray,
    **kwargs,
) -> list[GPE2DSpherePlateResult]:
    return [solve_sphere_plate(ch, d_min_m, float(R), **kwargs) for R in R_m_array]
