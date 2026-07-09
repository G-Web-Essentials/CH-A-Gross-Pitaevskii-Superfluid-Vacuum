#!/usr/bin/env python3
"""
CH black-hole exterior demo — spherical GPE through r_s with clock redshift.

Extends ch_gpe_gravity.py diagnostics:
  - ρ, Φ, g_eff vs r/r_s (multiple r_s/ξ)
  - Clock redshift models vs GR Schwarzschild reference
  - Join-region flag for v3 matched-BC solver

  python ch_gpe_bh_exterior_demo.py
  python ch_gpe_bh_exterior_demo.py --rs-hat-list 0.5 1.0 2.0 5.0
  python ch_gpe_bh_exterior_demo.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import C, G_MEAS, CHParams
from ch_clock_redshift_from_gpe import (
    analytic_schwarzschild_clock_profile,
    clock_profile_from_gravity,
    format_clock_report,
    plot_clock_profile,
)
from ch_gpe_gravity import (
    analyze_analytic_profile,
    calibrate_alpha_g_default,
    mass_from_rs_hat,
    solve_gravity_sm_v3,
)

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def plot_bh_exterior_fields(
    result,
    title: str,
    out_path: Path,
    *,
    r_join_hat: float | None = None,
) -> None:
    """ρ, Φ, g_eff vs r/r_s with horizon marker."""
    r = result.r_m
    rs = result.r_s_schwarzschild_m
    x = r / rs if rs > 0 else r
    phi = result.phi_total
    g_newton = G_MEAS * result.mass_kg / np.clip(r**2, 1e-60, None)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].semilogx(x, result.rho_norm, "b-", lw=1.5, label=r"$\rho/\rho_{\mathrm{in}}$ (GPE)")
    rho_ana = np.clip(1.0 - rs / np.clip(r, rs * 1.001, None), 0.0, None) ** 2
    axes[0].semilogx(x, rho_ana, "k--", alpha=0.5, lw=1.0, label=r"$(1-r_s/r)^2$")
    axes[0].axvline(1.0, color="gray", ls=":", lw=0.9)
    axes[0].set_ylabel(r"$\rho/\rho_{\mathrm{in}}$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].semilogx(x, phi / C**2, "g-", lw=1.5, label=r"$\Phi/c^2$ total")
    if result.phi_q is not None:
        axes[1].semilogx(x, result.phi_q / C**2, "c--", alpha=0.7, label=r"$Q$ channel")
        axes[1].semilogx(x, result.phi_hydro / C**2, "m--", alpha=0.7, label=r"hydro")
    phi_newton = -G_MEAS * result.mass_kg / np.clip(r, 1e-30, None)
    axes[1].semilogx(x, phi_newton / C**2, "k--", alpha=0.45, label=r"$-GM/r$")
    axes[1].axvline(1.0, color="gray", ls=":", lw=0.9)
    axes[1].set_ylabel(r"$\Phi / c^2$")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].loglog(x, np.abs(result.accel_m_s2), "r-", lw=1.5, label=r"$|a|$ GPE")
    axes[2].loglog(x, g_newton, "k--", alpha=0.5, label=r"$GM/r^2$")
    axes[2].loglog(x, result.g_eff, "m-", alpha=0.8, lw=1.2, label=r"$G_{\mathrm{eff}}(r)$")
    axes[2].axvline(1.0, color="gray", ls=":", lw=0.9)
    if r_join_hat is not None and rs > 0:
        axes[2].axvline(r_join_hat * result.xi_m / rs, color="orange", ls="-.", lw=0.9, alpha=0.7, label="join")
    axes[2].set_xlabel(r"$r / r_s$")
    axes[2].set_ylabel(r"$|a|$, $G_{\mathrm{eff}}$ [SI]")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multi_rs_density(
    results: list[tuple[float, object]],
    out_path: Path,
) -> None:
    """Overlay ρ(r/r_s) for several r_s/ξ."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rs_hat, res in results:
        rs = res.r_s_schwarzschild_m
        x = res.r_m / rs
        ax.semilogx(x, res.rho_norm, lw=1.5, label=rf"$r_s/\xi={rs_hat:g}$")
    x_ref = np.logspace(-0.5, 2.5, 500)
    ax.semilogx(x_ref, np.clip(1.0 - 1.0 / x_ref, 0.0, None) ** 2, "k--", alpha=0.5, label=r"$(1-1/x)^2$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"$r / r_s$")
    ax.set_ylabel(r"$\rho/\rho_{\mathrm{in}}$")
    ax.set_title("BH exterior density — multiple $r_s/\\xi$ (v3 matched BC)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_redshift_comparison(
    ch: CHParams,
    result,
    out_path: Path,
) -> None:
    """Compare CH clock models to GR at fixed r_s/ξ."""
    profile = clock_profile_from_gravity(result, label=result.solver)
    rs = result.r_s_schwarzschild_m
    x = result.r_m / rs

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "sqrt_rho": ("C0", "-"),
        "mu_local": ("C1", "-"),
        "phi_ch": ("C2", "-"),
        "gr_static": ("k", "--"),
    }
    for model in profile.model_names():
        color, ls = styles.get(model, ("gray", "-"))
        ax.semilogx(x, profile.redshift[model], lw=1.5, color=color, ls=ls, label=model)

    # Percent difference: sqrt_rho vs GR in weak field
    if "sqrt_rho" in profile.redshift and "gr_static" in profile.redshift:
        z_ch = profile.redshift["sqrt_rho"]
        z_gr = profile.redshift["gr_static"]
        mask = (x > 1.5) & (x < 50)
        if np.any(mask):
            rel = np.median(np.abs(z_ch[mask] - z_gr[mask]) / np.clip(z_gr[mask], 1e-30, None))
            ax.text(
                0.03,
                0.97,
                f"median |z_sqrt_rho - z_GR|/z_GR for 1.5 < r/rs < 50: {rel:.1%}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

    ax.axvline(1.0, color="gray", ls=":", lw=0.9)
    ax.set_xlabel(r"$r / r_s$")
    ax.set_ylabel(r"Redshift $z$")
    ax.set_title(rf"Clock redshift — $r_s/\xi={rs/result.xi_m:.2f}$, solver={result.solver}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def solve_v3_calibrated(ch: CHParams, r_s_hat: float, r_join_hat: float = 12.0):
    v3 = solve_gravity_sm_v3(ch, r_s_hat=r_s_hat, r_join_hat=r_join_hat)
    _, v3_cal = calibrate_alpha_g_default(v3)
    return v3_cal


def main() -> None:
    parser = argparse.ArgumentParser(description="CH BH exterior + clock redshift demo")
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument(
        "--rs-hat-list",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0],
        help="Schwarzschild radii in units of ξ",
    )
    parser.add_argument("--r-join-hat", type=float, default=12.0)
    parser.add_argument("--quick", action="store_true", help="Single r_s/ξ=1.0 only")
    args = parser.parse_args()

    if args.quick:
        args.rs_hat_list = [1.0]

    ch = CHParams(xi=args.xi, alpha_g=1.0)
    reports: list[str] = []
    multi: list[tuple[float, object]] = []

    print("=" * 72)
    print("CH BH EXTERIOR — spherical GPE + clock redshift")
    print("=" * 72)
    print(f"ξ = {ch.xi:.3e} m, r_join = {args.r_join_hat} ξ")

    for rs_hat in args.rs_hat_list:
        mass = mass_from_rs_hat(ch, rs_hat)
        print(f"\nSolving v3 matched BC: r_s/ξ = {rs_hat:g}, M = {mass:.4e} kg ...")
        v3_cal = solve_v3_calibrated(ch, rs_hat, r_join_hat=args.r_join_hat)
        multi.append((rs_hat, v3_cal))

        tag = f"rs{rs_hat:g}".replace(".", "p")
        plot_bh_exterior_fields(
            v3_cal,
            f"BH exterior v3 — $r_s/\\xi={rs_hat:g}$",
            OUTPUT / f"ch_bh_exterior_{tag}.png",
            r_join_hat=args.r_join_hat,
        )
        plot_redshift_comparison(ch, v3_cal, OUTPUT / f"ch_bh_redshift_{tag}.png")
        prof = clock_profile_from_gravity(v3_cal, label=f"v3 rs_hat={rs_hat}")
        reports.append(format_clock_report(prof))

    plot_multi_rs_density(multi, OUTPUT / "ch_bh_exterior_multi_rs.png")

    # Analytic reference at rs_hat=1
    rs_ref = args.rs_hat_list[0]
    mass_ref = mass_from_rs_hat(ch, rs_ref)
    ana = analyze_analytic_profile(ch, mass_ref)
    plot_bh_exterior_fields(
        ana,
        "Analytic $(1-r_s/r)^2$ hydrostatic",
        OUTPUT / "ch_bh_exterior_analytic.png",
    )
    ana_clock = analytic_schwarzschild_clock_profile(ch, mass_ref)
    plot_clock_profile(ana_clock, OUTPUT / "ch_bh_clock_analytic.png")

    report_path = OUTPUT / "ch_bh_exterior_report.txt"
    header = [
        "CH BH exterior report",
        f"xi_m = {ch.xi}",
        f"r_join_hat = {args.r_join_hat}",
        f"rs_hat_list = {args.rs_hat_list}",
        "",
        "Notes:",
        "  - r/r_s = 1 marks Schwarzschild radius (horizon in GR; CH exterior matched BC).",
        "  - Inner domain [0, r_join]: numerical GPE with S_M sink; outer: analytic tail.",
        "  - sqrt_rho clock proxy tracks depletion; phi_ch uses CH effective potential.",
        "  - mu_local (1-rho) does NOT match GR redshift sign — included for comparison only.",
        "",
    ]
    report_path.write_text("\n".join(header) + "\n".join(reports) + "\n")

    print(f"\nWrote {OUTPUT / 'ch_bh_exterior_multi_rs.png'}")
    print(f"Wrote {report_path}")
    for rs_hat in args.rs_hat_list:
        tag = f"rs{rs_hat:g}".replace(".", "p")
        print(f"Wrote {OUTPUT / f'ch_bh_exterior_{tag}.png'}")
        print(f"Wrote {OUTPUT / f'ch_bh_redshift_{tag}.png'}")


if __name__ == "__main__":
    main()
