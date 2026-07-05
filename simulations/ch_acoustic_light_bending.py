"""
CH acoustic light bending — ray trace in effective index n(r) from ρ(r).

Weak-field and numeric ray-tracing tools compare CH deflection angles Δθ(b)
to Schwarzschild GR for the same mass / r_s.

Index models (static spherical, v=0):
  gr_isotropic — n = 1 − r_s/r  (isotropic weak-field Schwarzschild; α → 4GM/bc²)
  sqrt_rho     — n = 1/√ρ  (acoustic slowdown in depletion; CH amplitude proxy)
  phi_ch       — n = √(1 + 2Φ_CH/c²)  from GPE effective potential
  linear_rho   — n = 1/(1 − r_s/r)  (inverse of √(ρ) tail with ρ=(1−r_s/r)²)

Integration:
  eikonal  — fast weak-field line integral (default)
  geodesic — Hamilton ray ODE (solve_ivp) cross-check

See docs/ch-universal-laws-plain-english.md §3.2 and chronos-hydrodynamics-paper §5.2.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_gravity import (
    GravityGPEResult,
    alpha_g_required_for_hydrostatic_newton,
    analyze_analytic_profile,
    calibrate_alpha_g_from_defect,
    effective_phi,
    mass_from_rs_hat,
    schwarzschild_depletion_profile,
    solve_gravity_sm_v3,
)


class IndexModel(str, Enum):
    GR_ISOTROPIC = "gr_isotropic"
    SQRT_RHO = "sqrt_rho"
    PHI_CH = "phi_ch"
    LINEAR_RHO = "linear_rho"


@dataclass
class DeflectionResult:
    """Deflection angle vs impact parameter for one index model."""

    model: str
    b_m: np.ndarray
    alpha_rad: np.ndarray
    method: str
    mass_kg: float
    r_s_m: float
    label: str = ""

    @property
    def alpha_arcsec(self) -> np.ndarray:
        return np.degrees(self.alpha_rad) * 3600.0


def schwarzschild_radius(mass_kg: float) -> float:
    return G_MEAS * mass_kg / C**2


def deflection_gr(mass_kg: float, b_m: float | np.ndarray) -> np.ndarray:
    """Photon deflection [rad] matching repo r_s = GM/c²: α = 2r_s/b = 2GM/(bc²)."""
    rs = schwarzschild_radius(mass_kg)
    b = np.asarray(b_m, dtype=float)
    return 2.0 * rs / np.clip(b, 1e-30, None)


def deflection_gr_full(mass_kg: float, b_m: float | np.ndarray) -> np.ndarray:
    """Full Schwarzschild deflection with r_s = GM/c²: α = 4r_s/b = 4GM/(bc²)."""
    return 2.0 * deflection_gr(mass_kg, b_m)


def deflection_newtonian(mass_kg: float, b_m: float | np.ndarray) -> np.ndarray:
    """Newtonian half-angle [rad]: α = 2GM/(bc²) = r_s/b."""
    rs = schwarzschild_radius(mass_kg)
    b = np.asarray(b_m, dtype=float)
    return rs / np.clip(b, 1e-30, None)


def make_index_function(
    model: IndexModel,
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    r_s_m: float,
    ch: CHParams,
    phi: np.ndarray | None = None,
) -> Callable[[float], float]:
    """Build n(r) interpolator from a radial GPE profile."""
    r = np.asarray(r_m, dtype=float)
    rho = np.clip(np.asarray(rho_norm, dtype=float), 1e-12, None)
    order = np.argsort(r)
    r = r[order]
    rho = rho[order]

    if phi is None:
        phi_arr = effective_phi(ch, r, rho)
    else:
        phi_arr = np.asarray(phi, dtype=float)[order]

    if model == IndexModel.GR_ISOTROPIC:
        n_tab = np.clip(1.0 - r_s_m / np.clip(r, r_s_m * 1.001, None), 1e-9, None)
    elif model == IndexModel.SQRT_RHO:
        n_tab = 1.0 / np.sqrt(rho)
    elif model == IndexModel.PHI_CH:
        n_tab = np.sqrt(np.clip(1.0 + 2.0 * phi_arr / C**2, 1e-12, None))
    elif model == IndexModel.LINEAR_RHO:
        linear = np.clip(1.0 - r_s_m / np.clip(r, r_s_m * 1.001, None), 1e-9, None)
        n_tab = 1.0 / linear
    else:
        raise ValueError(f"Unknown model {model}")

    def n_of_r(r_query: float) -> float:
        rq = float(max(r_query, r[0]))
        return float(np.interp(rq, r, n_tab, left=float(n_tab[0]), right=float(n_tab[-1])))

    return n_of_r


def _dn_dr(n_of_r: Callable[[float], float], r: float, h: float | None = None) -> float:
    h = h or max(1e-9 * r, 1e-12)
    return (n_of_r(r + h) - n_of_r(r - h)) / (2.0 * h)


def deflection_eikonal(
    b_m: float,
    n_of_r: Callable[[float], float],
    *,
    z_max_factor: float = 200.0,
    r_min: float = 1e-12,
) -> float:
    """
    Weak-field eikonal deflection [rad] for spherically symmetric n(r).

    α(b) = 4 ∫_0^∞ (1/n)(dn/dr)(b/r) dz   (straight-line / small-angle limit)
    """
    b = float(b_m)
    if b <= r_min:
        return np.nan

    z_max = z_max_factor * b

    def integrand(z: float) -> float:
        r = float(np.hypot(b, z))
        if r <= r_min:
            return 0.0
        n = n_of_r(r)
        dn = _dn_dr(n_of_r, r)
        return 4.0 * (1.0 / n) * dn * (b / r)

    alpha, _ = quad(integrand, 0.0, z_max, limit=200)
    return float(alpha)


def deflection_geodesic(
    b_m: float,
    n_of_r: Callable[[float], float],
    *,
    z_extent_factor: float = 80.0,
    r_min: float = 1e-12,
) -> float:
    """
    Numeric ray trace in (x, z) with d/ds(n t) = ∇n; return total deflection [rad].
    """
    b = float(b_m)
    if b <= r_min:
        return np.nan

    z0 = -z_extent_factor * b
    z1 = z_extent_factor * b
    x0, z_start = b, z0
    t0 = np.array([0.0, 1.0], dtype=float)

    def grad_n(pos: np.ndarray) -> np.ndarray:
        x, z = pos
        r = float(np.hypot(x, z))
        if r <= r_min:
            return np.zeros(2)
        dr = max(1e-9 * r, 1e-12)
        r_hat = pos / r
        dn_dr = _dn_dr(n_of_r, r, h=dr)
        return dn_dr * r_hat

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        pos = y[:2]
        t = y[2:4]
        n = n_of_r(float(np.hypot(pos[0], pos[1])))
        gn = grad_n(pos)
        gdot = float(np.dot(gn, t))
        dt = (gn - gdot * t) / n
        return np.concatenate([t, dt])

    y0 = np.concatenate([np.array([x0, z_start]), t0])
    sol = solve_ivp(
        rhs,
        (0.0, 2.0 * z_extent_factor * b),
        y0,
        rtol=1e-9,
        atol=1e-11,
        max_step=b / 5.0,
        events=[
            lambda s, y: y[1] - z1,
        ],
    )
    if not sol.success or len(sol.t) < 2:
        return np.nan
    t_final = sol.y[2:4, -1]
    t_final /= np.linalg.norm(t_final)
    alpha = float(np.arctan2(t_final[0], t_final[1]))
    return alpha


def scan_deflection(
    b_grid_m: np.ndarray,
    n_of_r: Callable[[float], float],
    *,
    method: str = "geodesic",
    mass_kg: float,
    model: str,
    label: str = "",
) -> DeflectionResult:
    rs = schwarzschild_radius(mass_kg)
    alphas = np.zeros_like(b_grid_m, dtype=float)
    for i, b in enumerate(b_grid_m):
        if method == "geodesic":
            alphas[i] = deflection_geodesic(b, n_of_r)
        else:
            alphas[i] = deflection_eikonal(b, n_of_r)
    return DeflectionResult(
        model=model,
        b_m=b_grid_m,
        alpha_rad=alphas,
        method=method,
        mass_kg=mass_kg,
        r_s_m=rs,
        label=label,
    )


def deflection_from_gravity_result(
    result: GravityGPEResult,
    b_grid_m: np.ndarray,
    models: list[IndexModel] | None = None,
    method: str = "geodesic",
    ch_phi: CHParams | None = None,
) -> list[DeflectionResult]:
    """Compute deflection curves from a GravityGPEResult."""
    if models is None:
        models = list(IndexModel)
    ch = ch_phi if ch_phi is not None else result.ch
    out: list[DeflectionResult] = []
    for model in models:
        n_of_r = make_index_function(
            model,
            result.r_m,
            result.rho_norm,
            result.r_s_schwarzschild_m,
            ch,
            phi=None if model == IndexModel.PHI_CH else result.phi_total,
        )
        out.append(
            scan_deflection(
                b_grid_m,
                n_of_r,
                method=method,
                mass_kg=result.mass_kg,
                model=model.value,
                label=result.solver,
            )
        )
    return out


def format_deflection_report(
    results: list[DeflectionResult],
    mass_kg: float,
    *,
    b_probe_m: float | None = None,
) -> str:
    rs = schwarzschild_radius(mass_kg)
    if b_probe_m is None:
        b_probe_m = 50.0 * rs
    lines = [
        f"Light bending report — M = {mass_kg:.4e} kg, r_s = {rs:.4e} m",
        f"Probe b = {b_probe_m:.4e} m ({b_probe_m/rs:.1f} r_s)",
        f"GR reference α = {deflection_gr(mass_kg, b_probe_m):.4e} rad "
        f"({np.degrees(deflection_gr(mass_kg, b_probe_m))*3600:.2f} arcsec); "
        f"full GR (4GM/bc²) = {deflection_gr_full(mass_kg, b_probe_m):.4e} rad",
        "",
    ]
    for res in results:
        i = int(np.argmin(np.abs(res.b_m - b_probe_m)))
        a = res.alpha_rad[i]
        gr = float(deflection_gr(mass_kg, b_probe_m))
        ratio = a / gr if gr > 0 else np.nan
        lines.append(
            f"  [{res.model}] method={res.method}  "
            f"α={a:.4e} rad ({np.degrees(a)*3600:.2f}\")  "
            f"α/α_GR={ratio:.4f}  ({res.label})"
        )
    return "\n".join(lines)


def plot_deflection_curves(
    results: list[DeflectionResult],
    mass_kg: float,
    out_path: Path,
    *,
    title: str = "Acoustic light bending",
) -> None:
    rs = schwarzschild_radius(mass_kg)
    b_over_rs = None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    styles = {
        "gr_isotropic": ("C0", "-"),
        "sqrt_rho": ("C1", "-"),
        "phi_ch": ("C2", "-"),
        "linear_rho": ("C3", "-"),
    }
    b_ref = np.logspace(np.log10(3.0 * rs), np.log10(500.0 * rs), 200)
    ax1.loglog(b_ref / rs, deflection_gr(mass_kg, b_ref), "k--", lw=1.5, label="GR")
    ax2.semilogx(b_ref / rs, np.ones_like(b_ref), "k--", lw=1.5, label="GR ratio=1")

    for res in results:
        color, ls = styles.get(res.model, ("gray", "-"))
        x = res.b_m / rs
        gr = deflection_gr(mass_kg, res.b_m)
        ax1.loglog(x, res.alpha_rad, color=color, ls=ls, lw=1.5, label=res.model)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = res.alpha_rad / gr
        ax2.semilogx(x, ratio, color=color, ls=ls, lw=1.5, label=res.model)

    ax1.set_xlabel(r"$b / r_s$")
    ax1.set_ylabel(r"Deflection $\alpha$ [rad]")
    ax1.set_title(title)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r"$b / r_s$")
    ax2.set_ylabel(r"$\alpha / \alpha_{\mathrm{GR}}$")
    ax2.set_title("Ratio to Schwarzschild")
    ax2.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ray_paths(
    n_of_r: Callable[[float], float],
    b_list_m: list[float],
    mass_kg: float,
    out_path: Path,
    *,
    model_name: str = "index",
    z_extent_factor: float = 40.0,
) -> None:
    """Plot ray trajectories in the lens plane for selected impact parameters."""
    rs = schwarzschild_radius(mass_kg)
    fig, ax = plt.subplots(figsize=(7, 8))

    for b in b_list_m:
        z0 = -z_extent_factor * b
        z1 = z_extent_factor * b
        y0 = np.array([b, z0, 0.0, 1.0])

        def grad_n(pos: np.ndarray) -> np.ndarray:
            x, z = pos
            r = float(np.hypot(x, z))
            if r <= 1e-12:
                return np.zeros(2)
            dr = max(1e-9 * r, 1e-12)
            r_hat = pos / r
            dn_dr = _dn_dr(n_of_r, r, h=dr)
            return dn_dr * r_hat

        def rhs(s: float, y: np.ndarray) -> np.ndarray:
            pos = y[:2]
            t = y[2:4]
            n = n_of_r(float(np.hypot(pos[0], pos[1])))
            gn = grad_n(pos)
            gdot = float(np.dot(gn, t))
            dt = (gn - gdot * t) / n
            return np.concatenate([t, dt])

        sol = solve_ivp(
            rhs,
            (0.0, 3.0 * z_extent_factor * b),
            y0,
            rtol=1e-9,
            atol=1e-11,
            max_step=b / 8.0,
            dense_output=True,
            events=[lambda s, y: y[1] - z1],
        )
        if not sol.success:
            continue
        s_end = sol.t[-1]
        s_plot = np.linspace(0.0, s_end, 800)
        traj = sol.sol(s_plot)
        ax.plot(traj[0] / rs, traj[1] / rs, lw=1.2, label=rf"$b={b/rs:.0f}\,r_s$")

    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", ls=":", lw=1.0)
    ax.add_patch(circle)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$x / r_s$")
    ax.set_ylabel(r"$z / r_s$")
    ax.set_title(f"Ray paths — {model_name}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


OUTPUT = Path(__file__).parent / "output"


def run_demo(
    xi_m: float = 50e-9,
    rs_hat: float = 1.0,
    method: str = "geodesic",
    quick: bool = False,
) -> str:
    OUTPUT.mkdir(exist_ok=True)
    ch = CHParams(xi=xi_m, alpha_g=1.0)
    mass = mass_from_rs_hat(ch, rs_hat)
    rs = schwarzschild_radius(mass)

    if quick:
        b_grid = np.geomspace(5.0 * rs, 200.0 * rs, 25)
        profiles = [("analytic", analyze_analytic_profile(ch, mass))]
    else:
        b_grid = np.geomspace(3.0 * rs, 500.0 * rs, 40)
        v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat)
        _, v3_cal = calibrate_alpha_g_from_defect(v3)
        profiles = [
            ("analytic", analyze_analytic_profile(ch, mass)),
            ("v3", v3_cal),
        ]

    all_reports: list[str] = []

    for tag, result in profiles:
        ch_phi = CHParams(
            xi=ch.xi,
            alpha_g=alpha_g_required_for_hydrostatic_newton(result.ch),
        )
        curves = deflection_from_gravity_result(
            result, b_grid, method=method, ch_phi=ch_phi
        )
        all_reports.append(format_deflection_report(curves, mass))
        plot_deflection_curves(
            curves,
            mass,
            OUTPUT / f"ch_light_bending_{tag}.png",
            title=rf"Light bending — {tag} ($r_s/\xi={rs_hat:g}$)",
        )

        # Ray path plot for gr_isotropic model on this profile
        n_gr = make_index_function(
            IndexModel.GR_ISOTROPIC,
            result.r_m,
            result.rho_norm,
            rs,
            ch_phi,
            phi=result.phi_total,
        )
        b_rays = [10.0 * rs, 30.0 * rs, 100.0 * rs]
        plot_ray_paths(
            n_gr,
            b_rays,
            mass,
            OUTPUT / f"ch_light_bending_rays_{tag}.png",
            model_name=f"gr_isotropic ({tag})",
        )

    # Eikonal vs geodesic cross-check at one b
    b_test = 20.0 * rs
    rho_prof = schwarzschild_depletion_profile(ch, mass, np.geomspace(rs * 1.01, 1000 * rs, 500))
    r_test = np.geomspace(rs * 1.01, 1000 * rs, 500)
    n_iso = make_index_function(
        IndexModel.GR_ISOTROPIC, r_test, rho_prof, rs, ch
    )
    a_eik = deflection_eikonal(b_test, n_iso)
    a_geo = deflection_geodesic(b_test, n_iso)
    a_gr = float(deflection_gr(mass, b_test))
    cross = (
        f"\nCross-check at b = {b_test/rs:.0f} r_s (analytic n=1-r_s/r):\n"
        f"  eikonal  α = {a_eik:.6e} rad ({np.degrees(a_eik)*3600:.3f}\")\n"
        f"  geodesic α = {a_geo:.6e} rad ({np.degrees(a_geo)*3600:.3f}\")\n"
        f"  GR       α = {a_gr:.6e} rad ({np.degrees(a_gr)*3600:.3f}\")\n"
        f"  eik/GR   = {a_eik/a_gr:.4f}, geo/GR = {a_geo/a_gr:.4f}\n"
    )
    all_reports.append(cross)

    report = "\n".join(all_reports)
    (OUTPUT / "ch_light_bending_report.txt").write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CH acoustic light bending demo")
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--rs-hat", type=float, default=1.0)
    parser.add_argument("--method", choices=("eikonal", "geodesic"), default="geodesic")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("CH ACOUSTIC LIGHT BENDING")
    print("=" * 72)
    report = run_demo(
        xi_m=args.xi,
        rs_hat=args.rs_hat,
        method=args.method,
        quick=args.quick,
    )
    print(report)
    print(f"Wrote {OUTPUT / 'ch_light_bending_report.txt'}")
    for name in ("analytic", "v3") if not args.quick else ("analytic",):
        print(f"Wrote {OUTPUT / f'ch_light_bending_{name}.png'}")
        print(f"Wrote {OUTPUT / f'ch_light_bending_rays_{name}.png'}")


if __name__ == "__main__":
    main()
