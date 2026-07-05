"""
Axisymmetric GP sphere-plate solver — reference for MATLAB ch_axisym_gpe_solve.m.

  python ch_axisym_gpe_solve.py
  python ch_axisym_gpe_solve.py --quick
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ch_gpe_core import solve_sphere_plate


@dataclass
class AxisymSol:
    r_hat: np.ndarray
    z_hat: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    j_sphere: np.ndarray
    d_min_hat: float
    R_hat: float
    mu: float


def gap_height(r_hat: np.ndarray | float, d_min_hat: float, R_hat: float) -> np.ndarray:
    r = np.asarray(r_hat, dtype=float)
    return d_min_hat + r**2 / (2.0 * R_hat)


def j_sphere_indices(r: np.ndarray, z: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Last grid index with z <= h(r) (sphere surface BC at j+1)."""
    nr = len(r)
    js = np.ones(nr, dtype=int)
    for i in range(nr):
        mask = z <= H[i] + 1e-12
        js[i] = max(int(np.max(np.where(mask)[0])), 2)
    return js


def axisym_laplacian(
    f: np.ndarray,
    r: np.ndarray,
    dr: float,
    dz: float,
    j_sphere: np.ndarray,
) -> np.ndarray:
    nr, nz = f.shape
    lap = np.zeros_like(f, dtype=float)
    for i in range(nr):
        js = j_sphere[i]
        for j in range(1, js):  # j=0 is plate (Dirichlet)
            if j >= js - 1:
                continue
            # z second derivative with psi=0 at j=0 and j=js
            f_lo = 0.0 if j == 1 else f[i, j - 1]
            f_hi = 0.0 if j == js - 1 else f[i, j + 1]
            d2_dz2 = (f_hi - 2.0 * f[i, j] + f_lo) / dz**2

            if i == 0:
                d2_dr2 = 2.0 * (f[1, j] - f[0, j]) / dr**2
                lap[i, j] = d2_dr2 + d2_dz2
            elif i == nr - 1:
                d2_dr2 = (f[i - 1, j] - 2.0 * f[i, j] + f[i - 1, j]) / dr**2  # Neumann-ish
                lap[i, j] = d2_dr2 + d2_dz2
            else:
                d2_dr2 = (f[i + 1, j] - 2.0 * f[i, j] + f[i - 1, j]) / dr**2
                df_dr = (f[i + 1, j] - f[i - 1, j]) / (2.0 * dr)
                lap[i, j] = d2_dr2 + df_dr / max(r[i], 1e-12) + d2_dz2
    return lap


def normalize_bulk(psi: np.ndarray, j_sphere: np.ndarray) -> np.ndarray:
    nr, nz = psi.shape
    vals: list[float] = []
    i0 = max(nr // 4, 0)
    i1 = max(3 * nr // 4, i0 + 1)
    for i in range(i0, i1):
        js = j_sphere[i]
        j_a = max(1, js // 4)
        j_b = max(j_a + 1, 3 * js // 4)
        vals.append(float(np.mean(np.abs(psi[i, j_a:j_b]) ** 2)))
    mid = float(np.mean(vals)) if vals else 1.0
    if mid > 1e-30:
        psi = psi / np.sqrt(mid)
    return enforce_bc(psi, j_sphere)


def enforce_bc(psi: np.ndarray, j_sphere: np.ndarray) -> np.ndarray:
    nr, nz = psi.shape
    out = np.zeros_like(psi)
    for i in range(nr):
        js = j_sphere[i]
        out[i, 0] = 0.0
        out[i, 1:js] = psi[i, 1:js]
        out[i, js:] = 0.0
    return out


def solve_axisym_gpe(
    d_min_hat: float,
    R_hat: float,
    *,
    nr: int = 120,
    nz: int = 160,
    n_iter: int = 15000,
    dt: float = 0.003,
    tol: float = 1e-8,
    r_max_hat: float | None = None,
) -> AxisymSol:
    rim_r = min(1.0, 0.2 * R_hat)
    if r_max_hat is None:
        r_max_hat = max(2.5, rim_r * 2.0 + 0.5)

    r = np.linspace(0.0, r_max_hat, nr)
    dr = r[1] - r[0]
    z_top = float(gap_height(r_max_hat, d_min_hat, R_hat))
    z = np.linspace(0.0, z_top, nz)
    dz = z[1] - z[0]
    H = gap_height(r, d_min_hat, R_hat)
    js_arr = j_sphere_indices(r, z, H)

    rho0 = np.zeros((nr, nz))
    for i in range(nr):
        hi = H[i]
        rho0[i, :] = np.tanh(np.clip(z, 0, None)) * np.tanh(np.clip(hi - z, 0, None))
    psi = np.sqrt(np.clip(rho0, 0, None)).astype(complex)
    psi = enforce_bc(psi, js_arr)
    psi = normalize_bulk(psi, js_arr)

    mu = 0.0
    for _ in range(n_iter):
        rho = np.abs(psi) ** 2
        lap = axisym_laplacian(psi, r, dr, dz, js_arr)
        hpsi = -0.5 * lap + (1.0 - rho) * psi
        mu_new = float(np.real(np.vdot(psi.ravel(), hpsi.ravel()) / np.vdot(psi.ravel(), psi.ravel())))
        psi_new = psi - dt * (hpsi - mu_new * psi)
        psi_new = enforce_bc(psi_new, js_arr)
        psi_new = normalize_bulk(psi_new, js_arr)
        if abs(mu_new - mu) < tol and float(np.max(np.abs(psi_new - psi))) < tol:
            psi = psi_new
            mu = mu_new
            break
        psi = psi_new
        mu = mu_new

    return AxisymSol(
        r_hat=r,
        z_hat=z,
        psi=psi,
        rho=np.abs(psi) ** 2,
        j_sphere=js_arr,
        d_min_hat=d_min_hat,
        R_hat=R_hat,
        mu=mu,
    )


def interp_rho(sol: AxisymSol, r_q: float, z_q: float) -> float:
    r, z, rho = sol.r_hat, sol.z_hat, sol.rho
    ir = int(np.clip(np.searchsorted(r, r_q), 1, len(r) - 2))
    iz = int(np.clip(np.searchsorted(z, z_q), 1, len(z) - 2))
    # bilinear
    r0, r1 = r[ir], r[ir + 1]
    z0, z1 = z[iz], z[iz + 1]
    tr = (r_q - r0) / (r1 - r0) if r1 > r0 else 0.0
    tz = (z_q - z0) / (z1 - z0) if z1 > z0 else 0.0
    return float(
        (1 - tr) * (1 - tz) * rho[ir, iz]
        + tr * (1 - tz) * rho[ir + 1, iz]
        + (1 - tr) * tz * rho[ir, iz + 1]
        + tr * tz * rho[ir + 1, iz + 1]
    )


def grad_at(sol: AxisymSol, r_q: float, z_q: float) -> tuple[float, float]:
    dr = sol.r_hat[1] - sol.r_hat[0]
    dz = sol.z_hat[1] - sol.z_hat[0]
    rho_p = interp_rho(sol, r_q + dr, z_q)
    rho_m = interp_rho(sol, r_q - dr, z_q)
    rho_zp = interp_rho(sol, r_q, z_q + dz)
    rho_zm = interp_rho(sol, r_q, z_q - dz)
    drho_dr = (rho_p - rho_m) / (2 * dr)
    drho_dz = (rho_zp - rho_zm) / (2 * dz)
    return drho_dr, drho_dz


def probes_from_sol(sol: AxisymSol, rim_r_hat: float = 1.0) -> dict[str, float]:
    d = sol.d_min_hat
    z_wall = sol.z_hat[1]
    _, gdz_w = grad_at(sol, 0.0, z_wall)
    r_rim = min(rim_r_hat, 0.2 * sol.R_hat)
    z_probe = min(max(1.0, 0.1 * d), 0.45 * d)
    gdr, gdz = grad_at(sol, r_rim, z_probe)
    if d > 0.2:
        _, gdz_m = grad_at(sol, 0.0, d / 2)
        grad_mid = abs(gdz_m)
    else:
        grad_mid = abs(gdz_w)
    return {
        "grad_wall": abs(gdz_w),
        "grad_mid": grad_mid,
        "grad_radial": abs(gdr),
        "grad_rim": float(np.hypot(gdr, gdz)),
    }


def main() -> None:
    from ch_dispersion_core import CHParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=50e-9)
    d_min = 100e-9
    d_hat = d_min / ch.xi
    R_list = np.geomspace(1e-6, 1e-4, 6 if args.quick else 12)

    print(f"d_hat={d_hat:.3f}, xi={ch.xi:.3e}")
    for R_m in R_list:
        R_hat = R_m / ch.xi
        sol = solve_axisym_gpe(d_hat, R_hat)
        pr = probes_from_sol(sol)
        tf = solve_sphere_plate(ch, d_min, R_m)
        rel = abs(pr["grad_radial"] - tf.grad_ratio_radial) / max(tf.grad_ratio_radial, 1e-30)
        print(
            f"R={R_m*1e6:.3f} um  gp_rad={pr['grad_radial']:.4e}  "
            f"tf_rad={tf.grad_ratio_radial:.4e}  err={100*rel:.1f}%"
        )


if __name__ == "__main__":
    main()
