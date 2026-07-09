"""
Strong-field CH gravity — Q+Phi where rho curves, and physical back-reaction vs r_s/xi.

Route G5d: probe join-layer curvature (not flat bulk core) and whether
physical EL Phi-feedback grows with r_s/xi without artificial boosting.

  python ch_gravity_sm_strongfield.py
  python ch_gravity_sm_strongfield.py --quick
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, CHParams
from ch_gpe_gravity import (
    alpha_g_required_for_hydrostatic_newton,
    calibrate_g0_matched,
    effective_phi,
    hydrostatic_coupling,
    mass_from_rs_hat,
    quantum_potential_radial,
    sm_sink_coefficient,
    solve_gravity_sm_v3,
)
from ch_gravity_lgrav_variational import lambda_g_from_alpha
from ch_gravity_sm_coupled import (
    coupling_coeff_el,
    inner_mass_deficit,
    solve_coupled_sm,
)

OUTPUT = Path(__file__).parent / "output"
R_JOIN_HAT = 12.0


@dataclass
class BandStats:
    name: str
    r_lo_hat: float
    r_hi_hat: float
    rho_mean: float
    drho_dr_max: float
    q_rms: float
    phi_h_rms: float
    q_over_phi: float


@dataclass
class StrongFieldPoint:
    rs_hat: float
    rho_join: float
    m_inner_v3_kg: float
    m_inner_cpl_kg: float
    delta_m_inner_frac: float
    newton_v3: float
    newton_cpl: float
    newton_cpl_q: float
    feedback_over_sink_max: float
    z_phi: float
    z_sqrt_rho: float
    z_gr: float
    z_phi_over_z_gr: float
    bands: list[BandStats]


def band_mask(r_hat: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (r_hat >= lo) & (r_hat <= hi)


def analyze_bands(
    ch: CHParams,
    r_hat: np.ndarray,
    rho_norm: np.ndarray,
    rs_hat: float,
) -> list[BandStats]:
    """Q and Phi_hydro in bulk, join transition, and exterior bands."""
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    r_m = r_hat * ch.xi
    phi_h = hydrostatic_coupling(ch_ref, rho_norm)
    q = quantum_potential_radial(ch, r_m, rho_norm)
    drho = np.abs(np.gradient(rho_norm, r_hat))

    bands_spec = [
        ("bulk", 0.05, 0.35 * R_JOIN_HAT),
        ("join", 0.85 * R_JOIN_HAT, R_JOIN_HAT),
        ("exterior", R_JOIN_HAT * 1.02, min(R_JOIN_HAT * 4.0, float(r_hat[-1]))),
    ]
    out: list[BandStats] = []
    for name, lo, hi in bands_spec:
        m = band_mask(r_hat, lo, hi)
        if int(np.sum(m)) < 3:
            continue
        q_r = float(np.sqrt(np.mean(q[m] ** 2)))
        p_r = float(np.sqrt(np.mean(phi_h[m] ** 2)))
        out.append(
            BandStats(
                name=name,
                r_lo_hat=lo,
                r_hi_hat=hi,
                rho_mean=float(np.mean(rho_norm[m])),
                drho_dr_max=float(np.max(drho[m])),
                q_rms=q_r,
                phi_h_rms=p_r,
                q_over_phi=q_r / max(p_r, 1e-99),
            )
        )
    return out


def clock_redshift_phi_ch(
    ch: CHParams,
    r_m: np.ndarray,
    rho_norm: np.ndarray,
    r_emit_m: float,
) -> float:
    """z with Φ(∞)=0 convention: ω = √(1+2Φ/c²), ω(∞)=1."""
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    phi = effective_phi(ch_ref, r_m, rho_norm)
    i_emit = int(np.argmin(np.abs(r_m - r_emit_m)))
    omega_emit = float(np.sqrt(max(1.0 + 2.0 * phi[i_emit] / C**2, 1e-30)))
    return 1.0 / omega_emit - 1.0


def clock_redshift_sqrt_rho(rho_emit: float) -> float:
    """ω ∝ √ρ with ω(∞)=1 at ρ=ρ_in."""
    return float(1.0 / np.sqrt(max(rho_emit, 1e-12)) - 1.0)


def clock_redshift_gr(r_emit_m: float, r_s_m: float) -> float:
    """Schwarzschild static redshift reference."""
    x = r_s_m / max(r_emit_m, r_s_m * 1.001)
    return float(1.0 / np.sqrt(1.0 - x) - 1.0)


def gp_feedback_over_sink(
    ch: CHParams,
    r_hat: np.ndarray,
    rho_norm: np.ndarray,
    phi_j_kg: np.ndarray,
    g0: float,
    sigma_hat: float,
) -> float:
    """max |(lambda_g Phi)/(c^2 rho_in) psi| / |gamma psi| in sink-dominated core."""
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    lam_g = lambda_g_from_alpha(ch, alpha_ref)
    coeff = coupling_coeff_el(ch, lam_g)
    gamma = sm_sink_coefficient(r_hat, g0, sigma_hat)
    psi = np.sqrt(np.clip(rho_norm, 1e-12, 1.0))
    feedback = np.abs(coeff * phi_j_kg * psi)
    sink = np.abs(gamma * psi)
    # Exclude join layer where Gaussian sink → 0 (ratio ill-defined).
    mask = (r_hat <= 0.5 * R_JOIN_HAT) & (sink > 0.01 * g0)
    if int(np.sum(mask)) < 3:
        return float("nan")
    ratio = feedback[mask] / sink[mask]
    return float(np.max(ratio))


def evaluate_point(
    ch: CHParams,
    g0: float,
    sigma_hat: float,
    rs_hat: float,
) -> StrongFieldPoint:
    v3 = solve_gravity_sm_v3(ch, r_s_hat=rs_hat, sigma_hat=sigma_hat)
    cpl = solve_coupled_sm(
        ch,
        g0,
        sigma_hat,
        rs_hat,
        coupling_scale=1.0,
        artificial_coupling=False,
        n_outer=25,
    )
    alpha_ref = alpha_g_required_for_hydrostatic_newton(ch)
    ch_ref = CHParams(xi=ch.xi, alpha_g=alpha_ref)
    phi_h = hydrostatic_coupling(ch_ref, v3.rho_norm)

    m_v3 = inner_mass_deficit(ch, v3.r_hat, v3.rho_norm, R_JOIN_HAT)
    m_cpl = inner_mass_deficit(ch, cpl.r_hat, cpl.rho_norm, R_JOIN_HAT)
    rho_join = max((1.0 - rs_hat / R_JOIN_HAT) ** 2, 1e-6)

    r_s_m = rs_hat * ch.xi
    r_emit = 3.0 * r_s_m
    i_emit = int(np.argmin(np.abs(v3.r_m - r_emit)))
    rho_emit = float(v3.rho_norm[i_emit])
    z_phi = clock_redshift_phi_ch(ch, v3.r_m, v3.rho_norm, r_emit)
    z_sqrt = clock_redshift_sqrt_rho(rho_emit)
    z_gr = clock_redshift_gr(r_emit, r_s_m)

    return StrongFieldPoint(
        rs_hat=rs_hat,
        rho_join=rho_join,
        m_inner_v3_kg=m_v3,
        m_inner_cpl_kg=m_cpl,
        delta_m_inner_frac=(m_cpl - m_v3) / max(abs(m_v3), 1e-30),
        newton_v3=cpl.newton_v3_ref,
        newton_cpl=cpl.newton_coupled,
        newton_cpl_q=cpl.newton_coupled_q,
        feedback_over_sink_max=gp_feedback_over_sink(
            ch, v3.r_hat, v3.rho_norm, phi_h, g0, sigma_hat
        ),
        z_phi=z_phi,
        z_sqrt_rho=z_sqrt,
        z_gr=z_gr,
        z_phi_over_z_gr=z_phi / max(z_gr, 1e-30),
        bands=analyze_bands(ch, v3.r_hat, v3.rho_norm, rs_hat),
    )


def format_report(points: list[StrongFieldPoint]) -> str:
    lines = [
        "CH strong-field / physical back-reaction (Route G5d)",
        "",
        "=== Physical EL feedback vs r_s/xi (no artificial boost) ===",
    ]
    for p in points:
        lines.append(
            f"  rs/xi={p.rs_hat:g}  rho_join={p.rho_join:.4f}  "
            f"dM/M={p.delta_m_inner_frac:+.3e}  "
            f"Phi-feed/sink_max={p.feedback_over_sink_max:.3e}  "
            f"N_v3={p.newton_v3:.4f}  N_cpl={p.newton_cpl:.4f}"
        )

    lines += ["", "=== Q + Phi_hydro by radial band (v3 profile) ==="]
    for p in points:
        lines.append(f"  --- r_s/xi = {p.rs_hat:g} ---")
        for b in p.bands:
            lines.append(
                f"    {b.name:8s} [{b.r_lo_hat:.1f},{b.r_hi_hat:.1f}]xi: "
                f"rho={b.rho_mean:.4f}  |drho/dr|_max={b.drho_dr_max:.3e}  "
                f"|Q|/|Phi|={b.q_over_phi:.3e}"
            )

    lines += ["", "=== Clock redshift z at r = 3 r_s (Phi(inf)=0) ==="]
    for p in points:
        lines.append(
            f"  rs/xi={p.rs_hat:g}: z_GR={p.z_gr:.4f}  z_sqrt_rho={p.z_sqrt_rho:.4f}  "
            f"z_Phi={p.z_phi:.4f}  ratio Phi/GR={p.z_phi_over_z_gr:.4f}"
        )

    lines += [
        "",
        "Reading:",
        "  alpha_G^hydro,ref is grain-derived; Z_Phi matches Poisson to Phi_hydro (identification, not N-fit).",
        "  Physical Phi-feedback / sink in core (r < 0.5 r_join) stays tiny at grain alpha_G.",
        "  Q peaks at join layer (max |drho/dr|); still |Q|/|Phi| << 1 vs hydrostatic channel.",
        "  N_hydro drifts below 1 as r_s/xi grows (steeper deficit); not a separate fit knob.",
    ]
    return "\n".join(lines)


def plot_results(points: list[StrongFieldPoint], out_png: Path) -> None:
    rs = [p.rs_hat for p in points]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.semilogy(rs, [p.feedback_over_sink_max for p in points], "o-", label=r"$\Phi$-feed$/\gamma$")
    ax.set_xlabel(r"$r_s/\xi$")
    ax.set_ylabel("max feedback / sink")
    ax.set_title("Physical GP back-reaction")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rs, [p.newton_v3 for p in points], "o-", label="v3")
    ax.plot(rs, [p.newton_cpl for p in points], "s--", label="coupled")
    ax.axhline(1.0, color="k", ls=":", lw=0.6)
    ax.set_xlabel(r"$r_s/\xi$")
    ax.set_ylabel(r"$N_{\mathrm{hydro}}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for band_name, color in [("bulk", "C0"), ("join", "C3"), ("exterior", "C2")]:
        vals = []
        for p in points:
            b = next((x for x in p.bands if x.name == band_name), None)
            vals.append(b.q_over_phi if b else np.nan)
        ax.semilogy(rs, vals, "o-", color=color, label=band_name)
    ax.set_xlabel(r"$r_s/\xi$")
    ax.set_ylabel(r"$|Q|/|\Phi_{\mathrm{hydro}}|$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rs, [p.z_sqrt_rho for p in points], "o-", label=r"$z_{\sqrt{\rho}}$")
    ax.plot(rs, [p.z_phi for p in points], "s-", label=r"$z_{\Phi}$")
    ax.plot(rs, [p.z_gr for p in points], "k--", label=r"$z_{\mathrm{GR}}$")
    ax.set_xlabel(r"$r_s/\xi$")
    ax.set_ylabel(r"$z$ at $3r_s$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(points: list[StrongFieldPoint], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rs_hat",
                "rho_join",
                "delta_m_inner_frac",
                "feedback_over_sink_max",
                "newton_v3",
                "newton_cpl",
                "z_phi",
                "z_sqrt_rho",
                "z_gr",
                "q_over_phi_join",
            ]
        )
        for p in points:
            bj = next((b for b in p.bands if b.name == "join"), None)
            w.writerow(
                [
                    p.rs_hat,
                    p.rho_join,
                    p.delta_m_inner_frac,
                    p.feedback_over_sink_max,
                    p.newton_v3,
                    p.newton_cpl,
                    p.z_phi,
                    p.z_sqrt_rho,
                    p.z_gr,
                    bj.q_over_phi if bj else "",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    g0 = calibrate_g0_matched(1.0, sigma_hat=args.sigma)[0]
    rs_vals = [0.5, 1.0, 2.0, 4.0, 8.0] if not args.quick else [0.5, 1.0, 2.0, 4.0]

    points = [evaluate_point(ch, g0, args.sigma, rs) for rs in rs_vals]
    report = format_report(points)
    print(report)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    txt = OUTPUT / "ch_gravity_sm_strongfield.txt"
    txt.write_text(report + "\n")
    write_csv(points, OUTPUT / "ch_gravity_sm_strongfield.csv")
    print(f"Wrote {txt}")
    if not args.no_plot:
        plot_results(points, OUTPUT / "ch_gravity_sm_strongfield.png")
        print(f"Wrote {OUTPUT / 'ch_gravity_sm_strongfield.png'}")


if __name__ == "__main__":
    main()
