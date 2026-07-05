"""
CH clock rate and gravitational redshift from GPE density profiles.

Chronos identification: local clocks count condensate phase evolution. This module
post-processes spherical gravity solutions and 1D Casimir-gap profiles with
several clock-rate proxies (not all equivalent — compare curves explicitly).

Models (ω normalized to bulk reference ω_ref = 1 at ρ → ρ_ref):
  sqrt_rho   — ω ∝ √ρ  (depletion slows phase; matches far-field Schwarzschild tail)
  mu_local   — ω ∝ 1 − ρ  (static GP nonlinear chemical-potential density term)
  phi_ch     — dτ/dt ∝ √(1 + 2Φ_CH/c²)  from CH effective potential
  gr_static  — dτ/dt = √(1 − r_s/r)  (Schwarzschild reference)

Redshift (emitter at r, observer at infinity):  z = ω(∞)/ω(r) − 1.

See docs/ch-universal-laws-plain-english.md §2.4.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_gpe_core import GPE1DResult, solve_casimir_gap
from ch_gpe_gravity import (
    GravityGPEResult,
    alpha_g_required_for_hydrostatic_newton,
    effective_phi,
    mass_from_rs_hat,
    schwarzschild_depletion_profile,
)


class ClockModel(str, Enum):
    SQRT_RHO = "sqrt_rho"
    MU_LOCAL = "mu_local"
    PHI_CH = "phi_ch"
    GR_STATIC = "gr_static"


@dataclass
class ClockProfile:
    """Clock rates and redshifts along a radial or 1D coordinate."""

    label: str
    coord_m: np.ndarray
    coord_label: str
    rho_norm: np.ndarray
    omega_norm: dict[str, np.ndarray]
    redshift: dict[str, np.ndarray]
    r_s_m: float | None = None
    reference_note: str = ""

    def model_names(self) -> list[str]:
        return list(self.omega_norm.keys())


def _normalize_omega(omega: np.ndarray, i_ref: int) -> np.ndarray:
    ref = float(omega[i_ref])
    if abs(ref) < 1e-30:
        return np.full_like(omega, np.nan)
    return omega / ref


def _redshift_from_omega(omega_norm: np.ndarray) -> np.ndarray:
    """z = ω(∞)/ω(r) − 1 with ω(∞) = 1 after normalization."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / np.clip(omega_norm, 1e-30, None) - 1.0
    return z


def omega_sqrt_rho(rho_norm: np.ndarray, rho_floor: float = 1e-6) -> np.ndarray:
    return np.sqrt(np.clip(rho_norm, rho_floor, None))


def omega_mu_local(rho_norm: np.ndarray, rho_ceil: float = 1.0 - 1e-6) -> np.ndarray:
    return np.clip(1.0 - np.clip(rho_norm, 0.0, rho_ceil), 1e-6, None)


def omega_phi_ch(ch: CHParams, r_m: np.ndarray, rho_norm: np.ndarray) -> np.ndarray:
    phi = effective_phi(ch, r_m, rho_norm)
    factor = np.sqrt(np.clip(1.0 + 2.0 * phi / C**2, 1e-30, None))
    return factor


def omega_gr_static(r_m: np.ndarray, r_s_m: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.sqrt(np.clip(1.0 - r_s_m / np.clip(r_m, r_s_m * 1.001, None), 0.0, None))
    return factor


def clock_models_for_sphere(
    ch: CHParams,
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    r_s_m: float,
    models: list[ClockModel] | None = None,
    i_ref: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute normalized ω and redshift z for spherical profiles."""
    if models is None:
        models = list(ClockModel)
    if i_ref is None:
        i_ref = len(r_m) - 1

    omega_raw: dict[str, np.ndarray] = {}
    if ClockModel.SQRT_RHO in models:
        omega_raw[ClockModel.SQRT_RHO.value] = omega_sqrt_rho(rho_norm)
    if ClockModel.MU_LOCAL in models:
        omega_raw[ClockModel.MU_LOCAL.value] = omega_mu_local(rho_norm)
    if ClockModel.PHI_CH in models:
        ch_phi = CHParams(
            xi=ch.xi,
            alpha_g=alpha_g_required_for_hydrostatic_newton(ch),
        )
        omega_raw[ClockModel.PHI_CH.value] = omega_phi_ch(ch_phi, r_m, rho_norm)
    if ClockModel.GR_STATIC in models:
        omega_raw[ClockModel.GR_STATIC.value] = omega_gr_static(r_m, r_s_m)

    omega_norm = {k: _normalize_omega(v, i_ref) for k, v in omega_raw.items()}
    redshift = {k: _redshift_from_omega(v) for k, v in omega_norm.items()}
    return omega_norm, redshift


def clock_profile_from_gravity(
    result: GravityGPEResult,
    label: str = "gravity",
    models: list[ClockModel] | None = None,
) -> ClockProfile:
    """Build clock profile from a GravityGPEResult."""
    i_ref = int(len(result.r_m) - 1)
    omega_norm, redshift = clock_models_for_sphere(
        result.ch,
        result.r_m,
        result.rho_norm,
        result.r_s_schwarzschild_m,
        models=models,
        i_ref=i_ref,
    )
    return ClockProfile(
        label=label,
        coord_m=result.r_m,
        coord_label=r"$r$ [m]",
        rho_norm=result.rho_norm,
        omega_norm=omega_norm,
        redshift=redshift,
        r_s_m=result.r_s_schwarzschild_m,
        reference_note=f"ω_ref at r = {result.r_m[i_ref]:.3e} m (outer grid point)",
    )


def clock_profile_from_gap(
    gap: GPE1DResult,
    models: list[ClockModel] | None = None,
) -> ClockProfile:
    """Clock profile along Casimir gap (1D); reference at mid-gap bulk."""
    if models is None:
        models = [ClockModel.SQRT_RHO, ClockModel.MU_LOCAL]

    z_m = gap.z_hat * gap.xi_m
    rho = gap.rho_norm
    i_ref = int(np.argmin(np.abs(gap.z_hat - gap.d_hat / 2.0)))
    # Hard-wall BC forces ρ→0 at plates; floor avoids unphysical ω divergence in plots.
    rho_floor = 1e-4

    omega_raw: dict[str, np.ndarray] = {}
    if ClockModel.SQRT_RHO in models:
        omega_raw[ClockModel.SQRT_RHO.value] = omega_sqrt_rho(rho, rho_floor=rho_floor)
    if ClockModel.MU_LOCAL in models:
        omega_raw[ClockModel.MU_LOCAL.value] = omega_mu_local(rho)

    omega_norm = {k: _normalize_omega(v, i_ref) for k, v in omega_raw.items()}
    redshift = {k: _redshift_from_omega(v) for k, v in omega_norm.items()}

    return ClockProfile(
        label=f"gap d={gap.d_m*1e9:.0f} nm",
        coord_m=z_m,
        coord_label=r"$z$ [m]",
        rho_norm=rho,
        omega_norm=omega_norm,
        redshift=redshift,
        r_s_m=None,
        reference_note=f"ω_ref at mid-gap z = {z_m[i_ref]:.3e} m",
    )


def analytic_schwarzschild_clock_profile(
    ch: CHParams,
    mass_kg: float,
    r_max_hat: float = 500.0,
    n_points: int = 4000,
) -> ClockProfile:
    """Clock profile on analytic (1 − r_s/r)² depletion ansatz."""
    r_hat = np.linspace(1.01, r_max_hat, n_points)
    r_m = r_hat * ch.xi
    rho = schwarzschild_depletion_profile(ch, mass_kg, r_m)
    r_s = G_MEAS * mass_kg / C**2
    omega_norm, redshift = clock_models_for_sphere(ch, r_m, rho, r_s)
    return ClockProfile(
        label="analytic (1−r_s/r)²",
        coord_m=r_m,
        coord_label=r"$r$ [m]",
        rho_norm=rho,
        omega_norm=omega_norm,
        redshift=redshift,
        r_s_m=r_s,
        reference_note="ω_ref at outer radius",
    )


def format_clock_report(profile: ClockProfile) -> str:
    lines = [
        f"Clock profile: {profile.label}",
        f"Reference: {profile.reference_note}",
    ]
    if profile.r_s_m is not None:
        lines.append(f"r_s = {profile.r_s_m:.4e} m")
    for model in profile.model_names():
        z = profile.redshift[model]
        r = profile.coord_m
        if profile.r_s_m is not None:
            i_rs = int(np.argmin(np.abs(r - 2.0 * profile.r_s_m)))
            i_10rs = int(np.argmin(np.abs(r - 10.0 * profile.r_s_m)))
            lines.append(
                f"  [{model}] z(2 r_s)={z[i_rs]:.4e}, z(10 r_s)={z[i_10rs]:.4e}, z_max={np.nanmax(z):.4e}"
            )
        else:
            lines.append(
                f"  [{model}] z_wall_max={np.nanmax(z):.4e}, z_mid={z[len(z)//2]:.4e}"
            )
    return "\n".join(lines)


def plot_clock_profile(
    profile: ClockProfile,
    out_path: Path,
    *,
    use_r_over_rs: bool = True,
) -> None:
    """Two-panel: ω/ω_ref and redshift z vs coordinate."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    if use_r_over_rs and profile.r_s_m is not None and profile.r_s_m > 0:
        x = profile.coord_m / profile.r_s_m
        xlabel = r"$r / r_s$"
        rs_line = 1.0
    else:
        x = profile.coord_m
        xlabel = profile.coord_label
        rs_line = None

    styles = {
        "sqrt_rho": ("C0", "-"),
        "mu_local": ("C1", "-"),
        "phi_ch": ("C2", "-"),
        "gr_static": ("k", "--"),
    }
    for model in profile.model_names():
        color, ls = styles.get(model, ("gray", "-"))
        axes[0].semilogx(x, profile.omega_norm[model], label=model, color=color, ls=ls, lw=1.5)
        axes[1].semilogx(x, profile.redshift[model], label=model, color=color, ls=ls, lw=1.5)

    if rs_line is not None:
        for ax in axes:
            ax.axvline(rs_line, color="gray", ls=":", lw=0.9, alpha=0.7)
            ax.axvline(rs_line, color="gray", ls=":", lw=0.9, alpha=0.7, label=r"$r=r_s$" if ax is axes[0] else None)

    axes[0].set_ylabel(r"$\omega / \omega_{\mathrm{ref}}$")
    axes[0].set_title(f"Clock rate — {profile.label}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel(r"Redshift $z = \omega_\infty/\omega - 1$")
    axes[1].set_xlabel(xlabel)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare_redshift_models_at_r(
    profile: ClockProfile,
    r_query_m: float,
) -> dict[str, float]:
    """Interpolate redshift at a given radius."""
    out: dict[str, float] = {}
    for model in profile.model_names():
        out[model] = float(np.interp(r_query_m, profile.coord_m, profile.redshift[model]))
    return out


OUTPUT = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="CH clock redshift from GPE profiles")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--rs-hat", type=float, default=1.0, help="Schwarzschild radius in ξ")
    parser.add_argument("--gap-nm", type=float, default=150.0, help="Casimir gap [nm] for 1D profile")
    parser.add_argument("--gravity", action="store_true", help="Use v3 gravity solve (slow)")
    args = parser.parse_args()

    OUTPUT.mkdir(exist_ok=True)
    ch = CHParams(xi=args.xi, alpha_g=1.0)
    mass = mass_from_rs_hat(ch, args.rs_hat)
    reports: list[str] = []

    ana = analytic_schwarzschild_clock_profile(ch, mass)
    reports.append(format_clock_report(ana))
    plot_clock_profile(ana, OUTPUT / "ch_clock_redshift_analytic.png")

    if args.gravity:
        from ch_gpe_gravity import calibrate_alpha_g_from_defect, solve_gravity_sm_v3

        v3 = solve_gravity_sm_v3(ch, r_s_hat=args.rs_hat)
        _, v3_cal = calibrate_alpha_g_from_defect(v3)
        grav = clock_profile_from_gravity(v3_cal, label="gravity v3 (α_G cal)")
        reports.append(format_clock_report(grav))
        plot_clock_profile(grav, OUTPUT / "ch_clock_redshift_gravity_v3.png")

    gap = solve_casimir_gap(ch, args.gap_nm * 1e-9)
    gap_prof = clock_profile_from_gap(gap)
    reports.append(format_clock_report(gap_prof))
    plot_clock_profile(gap_prof, OUTPUT / "ch_clock_redshift_gap.png", use_r_over_rs=False)

    report_path = OUTPUT / "ch_clock_redshift_report.txt"
    report_path.write_text("\n\n".join(reports) + "\n")

    print("=" * 72)
    print("CH CLOCK REDSHIFT — GPE post-processor")
    print("=" * 72)
    for block in reports:
        print(block)
        print()
    print(f"Wrote {OUTPUT / 'ch_clock_redshift_analytic.png'}")
    if args.gravity:
        print(f"Wrote {OUTPUT / 'ch_clock_redshift_gravity_v3.png'}")
    print(f"Wrote {OUTPUT / 'ch_clock_redshift_gap.png'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
