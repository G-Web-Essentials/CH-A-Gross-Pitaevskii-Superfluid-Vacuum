#!/usr/bin/env python3
"""
CH Kepler / vortex correspondence toy — quantized circulation vs smooth orbits.

CH sketch (chronos-hydrodynamics-paper §5.3): superfluid circulation
    Γ = ∮ v·dl = n h / m_grain,   n ∈ ℤ
Macroscopic bodies speculatively lock to vortex filaments in a point-defect
potential Φ = −GM/r. For large n, discrete circulation levels crowd together
and trajectories approach classical Keplerian ellipses.

This script does NOT derive orbits from the GPE; it is a correspondence demo:
  1. Quantized circular radii r_n from Γ = 2πr√(GM/r) = nκ, κ = h/m_grain
  2. Newtonian orbits with L_n = μ n h/(2π m_grain) in U_eff(r)
  3. n_eff along a classical path — shows planetary-scale n is huge (smooth limit)

  python ch_vortex_kepler_toy.py
  python ch_vortex_kepler_toy.py --xi 50e-9 --quick
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from ch_dispersion_core import C, G_MEAS, HBAR, M_EARTH, M_SUN, CHParams

OUTPUT = Path(__file__).parent / "output"
AU_M = 1.495_978_707e11


@dataclass
class OrbitResult:
    """Integrated 2D Kepler orbit."""

    label: str
    t_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    r_m: np.ndarray
    n_quantum: int
    angular_momentum: float
    energy_j: float
    period_s: float | None = None


def circulation_quantum(ch: CHParams) -> float:
    """κ = h/m_grain [m²/s]; Γ = nκ."""
    return HBAR * 2.0 * np.pi / ch.m_grain


def angular_momentum_quantized(mu_kg: float, n: int, ch: CHParams) -> float:
    """L_n = μ n ℏ / m_grain (equivalent to μ n h/(2π m_grain))."""
    return mu_kg * n * HBAR / ch.m_grain


def circular_radius_quantized(n: int, mass_central_kg: float, ch: CHParams) -> float:
    """r_n from 2πr√(GM/r) = nκ."""
    if n <= 0:
        return np.nan
    kappa = circulation_quantum(ch)
    return (n * kappa / (2.0 * np.pi)) ** 2 / (G_MEAS * mass_central_kg)


def n_from_angular_momentum(L_kg_m2_s: float, mu_kg: float, ch: CHParams) -> int:
    """n = L m_grain / (μ ℏ) from L = μ n ℏ / m_grain."""
    return int(round(L_kg_m2_s * ch.m_grain / (mu_kg * HBAR)))


def n_effective(ch: CHParams, r_m: np.ndarray, v_theta_m_s: np.ndarray) -> np.ndarray:
    """n = Γ m_grain / h along a path, Γ = 2π r v_θ."""
    h = 2.0 * np.pi * HBAR
    gamma = 2.0 * np.pi * r_m * v_theta_m_s
    return gamma * ch.m_grain / h


def kepler_period(mass_central_kg: float, semi_major_m: float) -> float:
    return 2.0 * np.pi * np.sqrt(semi_major_m**3 / (G_MEAS * mass_central_kg))


def _energy_from_state(mu_kg: float, r_m: float, vr: float, vt: float, mass_central_kg: float) -> float:
    return 0.5 * mu_kg * (vr**2 + vt**2) - G_MEAS * mass_central_kg * mu_kg / r_m


def integrate_orbit(
    mass_central_kg: float,
    mu_kg: float,
    L_kg_m2_s: float,
    r0_m: float,
    vr0_m_s: float,
    *,
    label: str = "orbit",
    n_quantum: int = 0,
    t_orbits: float = 1.5,
    n_points: int = 4000,
) -> OrbitResult:
    """Integrate planar Newtonian orbit with conserved L in polar coords."""
    vt0 = L_kg_m2_s / (mu_kg * r0_m)
    energy = _energy_from_state(mu_kg, r0_m, vr0_m_s, vt0, mass_central_kg)

    # Estimate period for time span (circular or vis-viva semi-major)
    if energy < 0:
        a = -G_MEAS * mass_central_kg * mu_kg / (2.0 * energy)
        t_max = t_orbits * kepler_period(mass_central_kg, a)
    else:
        t_max = t_orbits * 2.0 * np.pi * r0_m / max(abs(vt0), 1.0)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, yp, vx, vy = y
        r = float(np.hypot(x, yp))
        r = max(r, 1e-6)
        factor = G_MEAS * mass_central_kg / r**3
        ax = -factor * x
        ay = -factor * yp
        return np.array([vx, vy, ax, ay])

    y0 = np.array([r0_m, 0.0, vr0_m_s, vt0])
    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(rhs, (0.0, t_max), y0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Orbit integration failed: {sol.message}")

    x, y = sol.y[0], sol.y[1]
    r = np.hypot(x, y)
    period = None
    if energy < 0:
        period = kepler_period(mass_central_kg, -G_MEAS * mass_central_kg * mu_kg / (2.0 * energy))

    return OrbitResult(
        label=label,
        t_s=sol.t,
        x_m=x,
        y_m=y,
        r_m=r,
        n_quantum=n_quantum,
        angular_momentum=L_kg_m2_s,
        energy_j=energy,
        period_s=period,
    )


def classical_earth_orbit(ch: CHParams) -> OrbitResult:
    """Earth-like elliptical starter (slight eccentricity)."""
    r0 = AU_M
    L = M_EARTH * np.sqrt(G_MEAS * M_SUN * r0)
    vr0 = 500.0  # m/s radial kick → e ~ 0.016
    return integrate_orbit(
        M_SUN,
        M_EARTH,
        L,
        r0,
        vr0,
        label="classical Earth-like",
        n_quantum=-1,
        t_orbits=1.0,
    )


def orbit_at_quantum_n(
    n: int,
    mass_central_kg: float,
    mu_kg: float,
    ch: CHParams,
    *,
    vr_fraction: float = 0.0,
) -> OrbitResult:
    """Start on quantized circular radius with optional radial kick."""
    r0 = circular_radius_quantized(n, mass_central_kg, ch)
    L = angular_momentum_quantized(mu_kg, n, ch)
    v_circ = np.sqrt(G_MEAS * mass_central_kg / r0)
    vr0 = vr_fraction * v_circ
    return integrate_orbit(
        mass_central_kg,
        mu_kg,
        L,
        r0,
        vr0,
        label=f"n={n}",
        n_quantum=n,
        t_orbits=2.0,
    )


def fractional_ladder_spacing(n: int) -> float:
    """Δr/r between adjacent circular levels r ∝ n²: (r_{n+1}-r_n)/r_n."""
    if n <= 0:
        return np.nan
    return ((n + 1) ** 2 - n**2) / n**2


def plot_quantized_ladder(
    ch: CHParams,
    mass_central_kg: float,
    n_max: int,
    classical: OrbitResult | None,
    out_path: Path,
) -> None:
    ns = np.arange(1, n_max + 1)
    radii = np.array([circular_radius_quantized(int(n), mass_central_kg, ch) for n in ns])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(ns, radii / AU_M, "b.-", lw=1.2, markersize=4)
    if classical is not None:
        a = -G_MEAS * M_SUN * M_EARTH / (2.0 * classical.energy_j)
        ax1.axhline(a / AU_M, color="k", ls="--", alpha=0.6, label=rf"classical $a \approx {a/AU_M:.3f}$ AU")
    ax1.set_xlabel(r"Circulation quantum $n$")
    ax1.set_ylabel(r"Quantized circular $r_n$ [AU]")
    ax1.set_title(r"CH vortex ladder: $r_n = (n\kappa/2\pi)^2/(GM)$")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    spacing = np.array([fractional_ladder_spacing(int(n)) for n in ns])
    ax2.loglog(ns, spacing, "r.-", lw=1.2, markersize=4)
    ax2.loglog(ns, 2.0 / ns, "k--", alpha=0.5, label=r"$2/n$ asymptote")
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$(r_{n+1}-r_n)/r_n$")
    ax2.set_title("Level spacing → 0 as $n \\to \\infty$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_orbit_comparison(
    orbits: list[OrbitResult],
    out_path: Path,
    *,
    in_au: bool = True,
) -> None:
    scale = AU_M if in_au else 1.0
    xlab = r"$x$ [AU]" if in_au else r"$x$ [m]"
    ylab = r"$y$ [AU]" if in_au else r"$y$ [m]"

    fig, ax = plt.subplots(figsize=(7, 7))
    for orb in orbits:
        ax.plot(orb.x_m / scale, orb.y_m / scale, lw=1.2, label=orb.label)
    ax.plot(0, 0, "k+", ms=12, mew=2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("Quantized $L_n$ orbits vs classical (point defect $-GM/r$)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_n_eff_along_classical(ch: CHParams, classical: OrbitResult, out_path: Path) -> None:
    """Effective quantum number along a smooth classical path."""
    x, y = classical.x_m, classical.y_m
    vx = np.gradient(x, classical.t_s)
    vy = np.gradient(y, classical.t_s)
    r = np.hypot(x, y)
    # Azimuthal speed: v·θ̂ = (-y vx + x vy)/r
    v_theta = (-y * vx + x * vy) / np.clip(r, 1e-6, None)
    n_eff = n_effective(ch, r, v_theta)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.semilogy(classical.t_s / (365.25 * 24 * 3600), n_eff, "b-", lw=1.0)
    ax1.set_ylabel(r"$n_{\mathrm{eff}} = \Gamma m_{\mathrm{grain}}/h$")
    ax1.set_title("Effective circulation quantum along classical Earth-like orbit")
    ax1.grid(True, alpha=0.3)

    rel_var = (n_eff - np.median(n_eff)) / np.median(n_eff)
    ax2.plot(classical.t_s / (365.25 * 24 * 3600), rel_var, "g-", lw=1.0)
    ax2.set_xlabel("Time [yr]")
    ax2.set_ylabel(r"$(n_{\mathrm{eff}} - \langle n\rangle)/\langle n\rangle$")
    ax2.set_title(r"Relative variation $\ll 1$ $\Rightarrow$ orbit looks continuous")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_radii_vs_classical(
    classical: OrbitResult,
    quantized: list[OrbitResult],
    out_path: Path,
) -> None:
    """|r(t) - r_classical(t)| / r for different n."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t_yr = classical.t_s / (365.25 * 24 * 3600)
    for q in quantized:
        n_pts = min(len(q.t_s), len(classical.t_s))
        rel = np.abs(q.r_m[:n_pts] - classical.r_m[:n_pts]) / np.clip(classical.r_m[:n_pts], 1e-6, None)
        ax.semilogy(t_yr[:n_pts], rel, lw=1.0, label=q.label)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(r"$|r - r_{\mathrm{classical}}|/r_{\mathrm{classical}}$")
    ax.set_title("Small-$n$ orbits deviate; large-$n$ tracks classical")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def format_report(
    ch: CHParams,
    classical: OrbitResult,
    n_earth_eff: float,
    n_list: list[int],
) -> str:
    kappa = circulation_quantum(ch)
    a = -G_MEAS * M_SUN * M_EARTH / (2.0 * classical.energy_j)
    lines = [
        "CH Kepler / vortex correspondence toy",
        f"ξ = {ch.xi:.3e} m,  m_grain = {ch.m_grain:.3e} kg",
        f"κ = h/m_grain = {kappa:.3e} m²/s",
        "",
        "Quantized circular radius: r_n = (nκ/2π)² / (GM)",
        f"Angular momentum: L_n = μ n ℏ / m_grain",
        "",
        f"Earth-like classical semi-major a ≈ {a/AU_M:.4f} AU",
        f"Classical L ≈ {classical.angular_momentum:.3e} kg m²/s",
        f"Effective n along classical orbit: n_eff ≈ {n_earth_eff:.3e}",
        f"Fractional ladder spacing at that n: Δr/r ≈ {fractional_ladder_spacing(int(round(n_earth_eff))):.3e}",
        "",
        "Interpretation (correspondence only, not GPE derivation):",
        "  - Small n: discrete circulation levels → visibly distinct radii/orbits",
        "  - Large n: Δr/r ~ 2/n → 0; classical Keplerian motion recovered",
        f"  - Planetary scale: n ~ {n_earth_eff:.1e} → smooth orbits expected",
        "",
        "Sample quantized circular radii [AU]:",
    ]
    for n in n_list:
        r = circular_radius_quantized(n, M_SUN, ch)
        lines.append(f"  n={n:6d}  r_n={r/AU_M:.4e} AU")
    lines.append("")
    lines.append("Status: correspondence sketch; does not prove vortex locking.")
    return "\n".join(lines)


def run_demo(ch: CHParams, *, quick: bool = False) -> str:
    OUTPUT.mkdir(exist_ok=True)
    classical = classical_earth_orbit(ch)

    x, y = classical.x_m, classical.y_m
    vx = np.gradient(x, classical.t_s)
    vy = np.gradient(y, classical.t_s)
    r = np.hypot(x, y)
    v_theta = (-y * vx + x * vy) / np.clip(r, 1e-6, None)
    n_eff_median = float(np.median(n_effective(ch, r, v_theta)))

    # n matching Earth angular momentum scale
    L_earth = classical.angular_momentum
    n_earth = n_from_angular_momentum(L_earth, M_EARTH, ch)
    n_earth = max(n_earth, 1)

    if quick:
        n_small = [1, 2, 5]
        n_large = [max(n_earth // 1000, 10)]
    else:
        n_small = [1, 2, 5, 10]
        n_large = [max(n_earth // 100, 100), max(n_earth // 10, 1000)]

    n_max_ladder = 50 if quick else 200
    plot_quantized_ladder(
        ch, M_SUN, n_max_ladder, classical, OUTPUT / "ch_vortex_kepler_ladder.png"
    )
    plot_n_eff_along_classical(ch, classical, OUTPUT / "ch_vortex_kepler_n_eff.png")

    orbits_small = [orbit_at_quantum_n(n, M_SUN, M_EARTH, ch, vr_fraction=0.05) for n in n_small]
    plot_orbit_comparison(
        [classical] + orbits_small[:3],
        OUTPUT / "ch_vortex_kepler_orbits_small_n.png",
    )

    # Large-n orbit: same energy scale via matching n ≈ n_earth
    orb_large = orbit_at_quantum_n(n_earth, M_SUN, M_EARTH, ch, vr_fraction=0.05)
    orb_large.label = f"n≈n_Earth ({n_earth:.1e})"
    plot_orbit_comparison(
        [classical, orb_large],
        OUTPUT / "ch_vortex_kepler_orbits_large_n.png",
    )

    # Compare radii time series: small n vs classical (same time window, different ICs)
    plot_radii_vs_classical(
        classical,
        [orb_large],
        OUTPUT / "ch_vortex_kepler_r_deviation.png",
    )

    report = format_report(ch, classical, n_eff_median, n_small + [n_earth])
    (OUTPUT / "ch_vortex_kepler_report.txt").write_text(report + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CH Kepler / vortex correspondence toy")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi, alpha_g=1.0)
    print("=" * 72)
    print("CH KEPLER / VORTEX CORRESPONDENCE TOY")
    print("=" * 72)
    report = run_demo(ch, quick=args.quick)
    print(report)
    print(f"Wrote {OUTPUT / 'ch_vortex_kepler_ladder.png'}")
    print(f"Wrote {OUTPUT / 'ch_vortex_kepler_n_eff.png'}")
    print(f"Wrote {OUTPUT / 'ch_vortex_kepler_orbits_small_n.png'}")
    print(f"Wrote {OUTPUT / 'ch_vortex_kepler_orbits_large_n.png'}")
    print(f"Wrote {OUTPUT / 'ch_vortex_kepler_r_deviation.png'}")
    print(f"Wrote {OUTPUT / 'ch_vortex_kepler_report.txt'}")


if __name__ == "__main__":
    main()
