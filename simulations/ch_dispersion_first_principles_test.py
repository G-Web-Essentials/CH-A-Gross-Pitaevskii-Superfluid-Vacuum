#!/usr/bin/env python3
"""
CH dispersion test from first principles (Prediction #4).

Derives the maximum quadratic dispersion coefficient from CH/GPE parameters:

    E_ξ = ℏ c_s / ξ          (vacuum resonance energy)
    β_max = (3/2) ξ² / ℏ²    (Bogoliubov / EFT mapping; SI)

Gradient-gated effective dispersion (CH Prediction #7):

    β_eff = β_max · χ(|∇ρ|),   χ = smooth_step(|∇ρ| − |∇ρ|_c)
    |∇ρ|_c = ρ_in / ξ          (natural GPE scale)

Compares:
  • Naive always-on CH (β = β_max everywhere)  → falsifiable vs Fermi LAT
  • Gradient-gated CH (β_eff ≈ 0 in quiet void) → consistent with GRB nulls
  • Lab discriminant: β_eff vs experimental |∇ρ| knob (synthetic analysis curve)

Requirements:
  pip install numpy scipy matplotlib

  python ch_dispersion_first_principles_test.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import (
    BETA_FERMI_BOUND,
    C,
    CHParams,
    D_L_GRB_GPC,
    D_L_GRB_M,
    E_PHOTON_GEV,
    E_QG_FERMI_GEV,
    G_MEAS,
    M_EARTH,
    M_SUN,
    R_EARTH,
    SIGMA_T_FERMI_S,
    beta_bound_from_timing,
    beta_eff,
    build_scenarios,
    chi,
    delay_seconds,
    grad_rho_far_field,
    verdict_always_on,
    xi_crit_always_on,
)

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def run_text_report(ch: CHParams) -> dict:
    beta_lim = beta_bound_from_timing(
        SIGMA_T_FERMI_S, D_L_GRB_M, E_PHOTON_GEV[0], E_PHOTON_GEV[-1]
    )
    xi_crit = xi_crit_always_on()

    print("=" * 76)
    print("CH DISPERSION — FIRST-PRINCIPLES TEST (Prediction #4 + #7)")
    print("=" * 76)
    print(f"\nCH parameters at ξ = {ch.xi:.3e} m:")
    print(f"  ρ_in      = {ch.rho_in:.4e} kg/m³")
    print(f"  m_grain   = {ch.m_grain:.4e} kg")
    print(f"  E_ξ       = {ch.e_xi_gev:.4e} GeV")
    print(f"  |∇ρ|_c    = {ch.grad_rho_crit:.4e} kg/m⁴")
    print(f"  β_max     = {ch.beta_max:.4e}  (always-on ceiling)")
    print(f"\nExperimental references:")
    print(f"  β_Fermi   ≲ {BETA_FERMI_BOUND:.4e}  (E_QG,2 ~ {E_QG_FERMI_GEV:.0e} GeV)")
    print(f"  β_lim(GRB)≲ {beta_lim:.4e}  (σ_t={SIGMA_T_FERMI_S}s, D={D_L_GRB_GPC} Gpc)")
    print(f"  ξ_crit    = {xi_crit:.3e} m  (always-on β_max = β_Fermi)")

    # Always-on verdict
    print(f"\n--- A. Naive always-on CH (β = β_max everywhere) ---")
    print(f"  {verdict_always_on(ch.beta_max)}")
    if ch.beta_max > BETA_FERMI_BOUND:
        print(f"  → Any ξ ≳ ξ_crit is ruled out IF dispersion is not gradient-gated.")
        print(f"  → CH survives only with β_eff ≪ β_max in quiet vacuum (Prediction #7).")

    print(f"\n--- B. Gradient-gated CH: β_eff = β_max · χ(|∇ρ|) ---\n")
    print(f"{'Scenario':<38} {'|∇ρ| [kg/m⁴]':<14} {'χ':<8} {'β_eff':<12} {'Δt@31GeV':<12} Verdict")
    print("-" * 100)

    rows = []
    for sc in build_scenarios(ch):
        g = sc.grad_rho
        x = float(chi(g, ch.grad_rho_crit))
        be = ch.beta_max * x
        dt = delay_seconds(31.0, be, sc.distance_m)
        v = (
            "PASS — χ ≈ 0 (quiet vacuum)" if be <= max(BETA_FERMI_BOUND * 1e-3, 1e-30)
            else "PASS — below Fermi bound" if be <= BETA_FERMI_BOUND
            else f"FAIL — measurable dispersion ({sc.name})"
        )
        print(
            f"{sc.name:<38} {g:<14.3e} {x:<8.4f} {be:<12.3e} {dt:<12.3e} {v}"
        )
        rows.append((sc.name, g, x, be, dt, v))

    print(f"\n--- C. Discriminating test (what would falsify CH vs QFT?) ---")
    print("  QFT + GR:        β = 0 at all |∇ρ| (flat null).")
    print("  Naive CH:        β = β_max always → excluded for realistic ξ.")
    print("  Gradient CH:     β_eff ≈ 0 in void; turns on only near |∇ρ| ≳ ρ_in/ξ.")
    print("  Falsifiable lab: measure photon Δt vs a knob that raises |∇ρ| (Casimir gap,")
    print("                   tidal ∇Φ, mass proximity). QFT → flat; CH → threshold turn-on.")
    print("  GRB 090510 path: void-dominated → β_eff ~ 0 → null is the CORRECT CH prediction.")

    # ξ scan summary
    print(f"\n--- D. ξ scan: always-on vs gated (void) ---")
    for label, xi in [("1 nm", 1e-9), ("150 nm", 150e-9), ("1 mm", 1e-3), ("ξ_crit", xi_crit)]:
        p = CHParams(xi=xi)
        ao = "FAIL" if p.beta_max > BETA_FERMI_BOUND else "PASS"
        print(f"  ξ={label:>6}: β_max={p.beta_max:.2e}  always-on={ao}  gated-void=PASS (χ=0)")

    print("=" * 76)

    return {
        "ch": ch,
        "beta_lim": beta_lim,
        "xi_crit": xi_crit,
        "rows": rows,
    }


def plot_results(ch: CHParams) -> None:
    xi_arr = np.logspace(-12, 0, 400)  # m
    beta_max_arr = np.array([CHParams(xi=x).beta_max for x in xi_arr])
    beta_fermi = np.full_like(xi_arr, BETA_FERMI_BOUND)
    xi_crit = xi_crit_always_on()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: β_max vs ξ
    ax = axes[0, 0]
    ax.loglog(xi_arr * 1e3, beta_max_arr, "b-", lw=2, label=r"$\beta_{\rm max}=(3/2)\xi^2/\hbar^2$")
    ax.axhline(BETA_FERMI_BOUND, color="r", ls="--", lw=2, label=rf"Fermi bound $\beta \lesssim {BETA_FERMI_BOUND:.1e}$")
    ax.axvline(xi_crit * 1e3, color="gray", ls=":", label=rf"$\xi_{{\rm crit}}$={xi_crit:.1e} m")
    for x_mm, lab in [(1e-6, "1 nm"), (0.15, "150 nm"), (1, "1 mm")]:
        ax.axvline(x_mm, color="gray", ls=":", alpha=0.4)
        ax.text(x_mm, 1e10, lab, rotation=90, fontsize=7, va="bottom")
    ax.fill_between(
        xi_arr * 1e3, beta_max_arr, beta_fermi,
        where=beta_max_arr > BETA_FERMI_BOUND,
        alpha=0.2, color="red", label="Always-on excluded",
    )
    ax.set_xlabel(r"Healing length $\xi$ (mm)")
    ax.set_ylabel(r"$\beta_{\rm max}$ (SI)")
    ax.set_title("First-principles CH: always-on dispersion vs Fermi")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # Panel B: β_eff vs |∇ρ|/|∇ρ|_c (log scale)
    ax = axes[0, 1]
    grad_ratio = np.logspace(-22, 2, 300)
    grad = grad_ratio * ch.grad_rho_crit
    be = beta_eff(grad, ch)
    ax.loglog(grad_ratio, np.maximum(be, 1e-20), "b-", lw=2, label=r"$\beta_{\rm eff}=\beta_{\rm max}\,\chi$")
    ax.axhline(BETA_FERMI_BOUND, color="r", ls="--", label="Fermi bound")
    ax.axvline(1.0, color="gray", ls=":", label=r"$|\nabla\rho|/|\nabla\rho|_c=1$")

    # Mark astrophysical points (in |∇ρ|/|∇ρ|_c units)
    marks = [
        ("Void", 0.0),
        ("Earth", grad_rho_far_field(ch.rho_in, M_EARTH, R_EARTH) / ch.grad_rho_crit),
        ("NS 12km", grad_rho_far_field(ch.rho_in, 1.4 * M_SUN, 12e3) / ch.grad_rho_crit),
        ("10 r_s", grad_rho_far_field(ch.rho_in, M_SUN, 10 * 2 * G_MEAS * M_SUN / C**2) / ch.grad_rho_crit),
    ]
    for name, ratio in marks:
        be_m = ch.beta_max * float(chi(ratio * ch.grad_rho_crit, ch.grad_rho_crit))
        ax.plot(ratio, max(be_m, 1e-50), "ko", ms=6)
        ax.annotate(name, (ratio, max(be_m, 1e-50)), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(r"$|\nabla\rho| \,/\, |\nabla\rho|_c$")
    ax.set_ylabel(r"$\beta_{\rm eff}$ (SI)")
    ax.set_title(f"Gradient-gated dispersion (ξ={ch.xi*1e3:.3g} mm)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel C: predicted GRB delay vs energy (void vs threshold-on)
    ax = axes[1, 0]
    e_grid = np.logspace(0, 2.5, 50)
    for label, beta, ls in [
        ("QFT null (β=0)", 0.0, ":"),
        ("CH void (β_eff≈0)", 0.0, "-"),
        (rf"CH at |∇ρ|_c (β={ch.beta_max:.1e})", ch.beta_max, "-"),
        ("Fermi exclude if β≳ bound", BETA_FERMI_BOUND, "--"),
    ]:
        dt = delay_seconds(e_grid, beta, D_L_GRB_M)
        ax.loglog(e_grid, np.maximum(dt, 1e-9), ls=ls, lw=2, label=label)
    ax.axhline(SIGMA_T_FERMI_S, color="gray", ls=":", label=rf"$\sigma_t \sim {SIGMA_T_FERMI_S}$ s")
    ax.set_xlabel("Photon energy (GeV)")
    ax.set_ylabel(r"Predicted $\Delta t$ (s)")
    ax.set_title(f"GRB 090510 scale (D={D_L_GRB_GPC} Gpc)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")

    # Panel D: lab discriminant — β_eff vs gradient knob (synthetic)
    ax = axes[1, 1]
    knob = np.logspace(-3, 1, 100)  # |∇ρ|/|∇ρ|_c proxy
    grad_lab = knob * ch.grad_rho_crit
    be_lab = beta_eff(grad_lab, ch)
    dt_lab = delay_seconds(31.0, be_lab, 10.0)  # 10 m baseline lab path

    ax2 = ax.twinx()
    ax.loglog(knob, np.maximum(be_lab, 1e-20), "b-", lw=2, label=r"CH $\beta_{\rm eff}$")
    ax2.loglog(knob, np.maximum(dt_lab * 1e12, 1e-6), "g--", lw=1.5, label=r"$\Delta t$ @ 31 GeV, L=10 m (ps)")
    ax.axhline(BETA_FERMI_BOUND, color="r", ls=":", lw=1, label="Fermi β bound")
    ax.axvline(1.0, color="gray", ls=":", label=r"$|\nabla\rho| = |\nabla\rho|_c$")
    ax.set_xlabel(r"Lab knob → $|\nabla\rho|/|\nabla\rho|_c$ (proxy)")
    ax.set_ylabel(r"$\beta_{\rm eff}$ (SI)", color="b")
    ax2.set_ylabel(r"$\Delta t$ (ps)", color="g")
    ax.set_title("Discriminant: CH threshold vs QFT flat null")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=6, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "CH dispersion first-principles test: β(ξ), gradient gate, and falsifiability",
        fontsize=12,
        y=1.01,
    )
    out = OUTPUT / "ch_dispersion_first_principles_test.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # Save numeric summary
    summary = OUTPUT / "ch_dispersion_first_principles_test.txt"
    with summary.open("w") as f:
        f.write(f"xi_m={ch.xi}\n")
        f.write(f"rho_in={ch.rho_in}\n")
        f.write(f"e_xi_gev={ch.e_xi_gev}\n")
        f.write(f"beta_max={ch.beta_max}\n")
        f.write(f"grad_rho_crit={ch.grad_rho_crit}\n")
        f.write(f"beta_fermi_bound={BETA_FERMI_BOUND}\n")
        f.write(f"xi_crit_always_on_m={xi_crit}\n")
        for sc in build_scenarios(ch):
            g = sc.grad_rho
            x = float(chi(g, ch.grad_rho_crit))
            be = ch.beta_max * x
            dt = delay_seconds(31.0, be, sc.distance_m)
            v = (
                "PASS — χ ≈ 0 (quiet vacuum)" if be <= max(BETA_FERMI_BOUND * 1e-3, 1e-30)
                else "PASS — below Fermi bound" if be <= BETA_FERMI_BOUND
                else f"FAIL — measurable dispersion ({sc.name})"
            )
            f.write(f"scenario={sc.name}|grad={g}|chi={x}|beta_eff={be}|dt31gev={dt}|{v}\n")
    print(f"Saved {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CH first-principles dispersion test")
    parser.add_argument(
        "--xi",
        type=float,
        default=1e-3,
        help="Healing length ξ in meters (default: 1 mm)",
    )
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    run_text_report(ch)
    plot_results(ch)


if __name__ == "__main__":
    main()
