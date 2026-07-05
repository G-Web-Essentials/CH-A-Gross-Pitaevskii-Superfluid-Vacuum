#!/usr/bin/env python3
"""
Combined CH dispersion test: first-principles β(ξ) + real Fermi GRB 090510 limit.

Integrates:
  • β_max = (3/2) ξ² / ℏ²  from CH/GPE
  • Gradient-gated β_eff on astrophysical paths
  • Fermi LAT literature bound (E_QG,2 ~ 10¹⁰–10¹¹ GeV)
  • Real GBM β_95 from fermi_grb090510_beta_limit.py (if available)
  • Real LAT LLE β_95 from fermi_grb090510_lat_lle_beta_limit.py (if available)

  python ch_dispersion_fermi_combined.py
  python ch_dispersion_fermi_combined.py --xi 1e-3
  python ch_dispersion_fermi_combined.py --run-fermi   # download + fit first
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import (
    BETA_FERMI_BOUND,
    BETA_FERMI_LO,
    CHParams,
    D_L_GRB_GPC,
    D_L_GRB_M,
    E_PHOTON_GEV,
    E_QG_FERMI_GEV,
    E_QG_FERMI_LO_GEV,
    SIGMA_T_FERMI_S,
    Z_GRB,
    beta_bound_from_timing,
    beta_eff,
    build_scenarios,
    chi,
    delay_seconds,
    e_qg_from_beta,
    grad_rho_far_field,
    verdict_always_on,
    verdict_gated,
    xi_crit_always_on,
)

OUTPUT = Path(__file__).parent / "output"
FERMI_GBM_TXT = OUTPUT / "fermi_grb090510_beta_real.txt"
FERMI_LLE_TXT = OUTPUT / "fermi_grb090510_lat_lle_beta_real.txt"
FERMI_EXT_TXT = OUTPUT / "fermi_grb090510_lat_extended_beta_real.txt"
FERMI_GBM_SCRIPT = Path(__file__).parent / "fermi_grb090510_beta_limit.py"
FERMI_LLE_SCRIPT = Path(__file__).parent / "fermi_grb090510_lat_lle_beta_limit.py"
FERMI_EXT_SCRIPT = Path(__file__).parent / "fermi_grb090510_lat_extended_beta_limit.py"


def load_result_txt(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                data[k] = float(v)
            except ValueError:
                pass
    return data if "beta_95" in data else None


def load_fermi_results() -> dict[str, float] | None:
    return load_result_txt(FERMI_GBM_TXT)


def load_lle_results() -> dict[str, float] | None:
    return load_result_txt(FERMI_LLE_TXT)


def load_extended_results() -> dict[str, float] | None:
    return load_result_txt(FERMI_EXT_TXT)


def run_fermi_pipeline() -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float] | None]:
    gbm, lle, ext = None, None, None
    if FERMI_GBM_SCRIPT.exists():
        print("Running fermi_grb090510_beta_limit.py ...")
        subprocess.run([sys.executable, str(FERMI_GBM_SCRIPT), "--no-download"], check=False)
        gbm = load_fermi_results()
    if FERMI_LLE_SCRIPT.exists():
        print("Running fermi_grb090510_lat_lle_beta_limit.py ...")
        subprocess.run([sys.executable, str(FERMI_LLE_SCRIPT), "--no-download"], check=False)
        lle = load_lle_results()
    if FERMI_EXT_SCRIPT.exists():
        print("Running fermi_grb090510_lat_extended_beta_limit.py ...")
        subprocess.run([sys.executable, str(FERMI_EXT_SCRIPT), "--no-download"], check=False)
        ext = load_extended_results()
    return gbm, lle, ext


def plot_combined(
    ch: CHParams,
    gbm: dict[str, float] | None,
    lle: dict[str, float] | None,
    ext: dict[str, float] | None,
) -> Path:
    xi_arr = np.logspace(-12, 0, 400)
    beta_max_arr = np.array([CHParams(xi=x).beta_max for x in xi_arr])
    xi_crit = xi_crit_always_on()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- A: β vs ξ with all experimental lines ---
    ax = axes[0, 0]
    ax.loglog(xi_arr * 1e3, beta_max_arr, "b-", lw=2.5, label=r"CH $\beta_{\rm max}=(3/2)\xi^2/\hbar^2$")
    ax.fill_between(
        xi_arr * 1e3, BETA_FERMI_LO, BETA_FERMI_BOUND,
        color="#ffcccc", alpha=0.7, label=rf"Fermi LAT ($E_{{QG,2}}\sim10^{{10}}$–$10^{{11}}$ GeV)",
    )
    ax.axhline(BETA_FERMI_BOUND, color="r", ls="--", lw=1.5)
    ax.axhline(BETA_FERMI_LO, color="r", ls=":", lw=1.5)

    beta_gbm = gbm["beta_95"] if gbm else None
    beta_lle = lle["beta_95"] if lle else None
    beta_ext = ext["beta_95"] if ext else None
    if beta_gbm is not None:
        ax.axhline(
            beta_gbm, color="#e67e22", ls="-.", lw=2,
            label=rf"Real GBM $\beta_{{95}}={beta_gbm:.1e}$",
        )
    if beta_lle is not None:
        ax.axhline(
            beta_lle, color="#2ca02c", ls="-.", lw=2,
            label=rf"LAT LLE $\beta_{{95}}={beta_lle:.1e}$",
        )
    if beta_ext is not None:
        ax.axhline(
            beta_ext, color="#9467bd", ls="-.", lw=2,
            label=rf"LAT extended $\beta_{{95}}={beta_ext:.1e}$",
        )

    ax.fill_between(
        xi_arr * 1e3, beta_max_arr, BETA_FERMI_BOUND,
        where=beta_max_arr > BETA_FERMI_BOUND,
        alpha=0.15, color="red", label="Naive always-on excluded",
    )
    ax.axhline(1e-30, color="green", ls="-", lw=3, alpha=0.6, label=r"Gradient CH void ($\beta_{\rm eff}\!\approx\!0$)")

    for x_mm, lab in [(1e-6, "1 nm"), (0.15, "150 nm"), (1, "1 mm")]:
        ax.axvline(x_mm, color="gray", ls=":", alpha=0.45)
        ax.text(x_mm, 1e8, lab, rotation=90, fontsize=7, va="bottom", color="gray")

    ax.axvline(xi_crit * 1e3, color="purple", ls=":", lw=1.5, label=rf"$\xi_{{\rm crit}}$={xi_crit:.1e} m")
    ax.set_xlabel(r"Healing length $\xi$ (mm)")
    ax.set_ylabel(r"$\beta$ (SI)")
    ax.set_title("CH β(ξ) vs Fermi limits\n(naive always-on fails; gated void passes)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-10, 1e65)

    # --- B: GRB path scenarios (gradient-gated) ---
    ax = axes[0, 1]
    scenarios = build_scenarios(ch)[:6]  # astro only
    names = [s.name for s in scenarios]
    betas = [float(ch.beta_max * chi(s.grad_rho, ch.grad_rho_crit)) for s in scenarios]
    colors = ["#2ca02c" if verdict_gated(b) == "PASS (χ≈0)" else "#d62728" for b in betas]
    ypos = np.arange(len(names))
    ax.barh(ypos, np.maximum(betas, 1e-35), color=colors, alpha=0.85)
    ax.axvline(BETA_FERMI_BOUND, color="r", ls="--", lw=2, label="Fermi LAT bound")
    if beta_gbm:
        ax.axvline(beta_gbm, color="#e67e22", ls="-.", lw=2, label="GBM β₉₅")
    if beta_lle:
        ax.axvline(beta_lle, color="#2ca02c", ls="-.", lw=2, label="LLE β₉₅")
    if beta_ext:
        ax.axvline(beta_ext, color="#9467bd", ls="-.", lw=2, label="Extended β₉₅")
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"$\beta_{\rm eff}$ (SI)")
    ax.set_title(f"Gradient-gated β on GRB paths (ξ={ch.xi*1e3:g} mm)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    # --- C: Real Fermi data summary + CH prediction ---
    ax = axes[1, 0]
    ax.axis("off")
    lines = [
        "GRB 090510 (z=0.903) — dispersion verdict",
        "─" * 52,
        f"CH β_max (ξ={ch.xi:.0e} m):     {ch.beta_max:.3e}",
        f"CH β_eff (void path):           ≈ 0  (χ → 0)",
        f"Naive always-on:                {verdict_always_on(ch.beta_max)}",
        f"Gradient-gated (void):          PASS — null expected",
        "",
        "Experimental bounds (SI β):",
        f"  Fermi LAT literature:          ≲ {BETA_FERMI_BOUND:.2e}",
        f"    (E_QG,2 ~ {E_QG_FERMI_LO_GEV:.0e}–{E_QG_FERMI_GEV:.0e} GeV)",
    ]
    if gbm:
        lines += [
            "",
            "Real Fermi GBM (MeV):",
            f"  Photons:                       {int(gbm.get('n_photons', 0))}",
            f"  β_95:                          {gbm['beta_95']:.2e}",
            f"  E_QG,2 mapped:                 ≳ {gbm.get('e_qg_gev', 0):.0f} GeV",
            f"  Fit: r={gbm.get('r_value', 0):.2f}, p={gbm.get('p_value', 0):.2f}",
        ]
    if lle:
        lines += [
            "",
            "Real Fermi LAT LLE (GeV):",
            f"  Photons:                       {int(lle.get('n_photons', 0))}",
            f"  β_95:                          {lle['beta_95']:.2e}",
            f"  E_QG,2 mapped:                 ≳ {lle.get('e_qg_gev', 0):.2e} GeV",
        ]
    if ext:
        lines += [
            "",
            "Real Fermi LAT extended (GeV, ~30 GeV photon):",
            f"  Photons:                       {int(ext.get('n_photons', 0))}",
            f"  β_95:                          {ext['beta_95']:.2e}",
            f"  E_QG,2 mapped:                 ≳ {ext.get('e_qg_gev', 0):.2e} GeV",
            f"  Fit: r={ext.get('r_value', 0):.2f}, p={ext.get('p_value', 0):.2f}",
        ]
    if not gbm and not lle and not ext:
        lines += [
            "",
            "Real Fermi data: not found.",
            f"Run: python {FERMI_GBM_SCRIPT.name}",
            f"     python {FERMI_LLE_SCRIPT.name}",
            f"     python {FERMI_EXT_SCRIPT.name}",
        ]

    lines += [
        "",
        "Falsifiable test (not GRB timing alone):",
        "  Measure photon Δt vs lab |∇ρ| knob.",
        "  QFT → flat; CH → threshold at |∇ρ|_c = ρ_in/ξ.",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
        fontsize=9, va="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="#ccc"),
    )

    # --- D: Predicted delay vs energy ---
    ax = axes[1, 1]
    e_grid = np.logspace(0, 2.5, 60)
    d_l = (gbm or lle or ext or {}).get("d_l_m", D_L_GRB_M)
    for label, beta, col, ls in [
        ("QFT / CH void (β≈0)", 0.0, "k", ":"),
        ("CH at |∇ρ|_c", ch.beta_max * 0.5, "b", "-"),
        ("Fermi exclude (β bound)", BETA_FERMI_BOUND, "r", "--"),
    ]:
        dt = delay_seconds(e_grid, beta, d_l)
        ax.loglog(e_grid, np.maximum(dt, 1e-12), ls=ls, lw=2, color=col, label=label)
    ax.axhline(SIGMA_T_FERMI_S, color="gray", ls=":", label=rf"Timing $\sigma_t\sim{SIGMA_T_FERMI_S}$ s")
    ax.set_xlabel("Photon energy (GeV)")
    ax.set_ylabel(r"Predicted $\Delta t$ (s)")
    ax.set_title(f"Arrival delay at D={d_l/3.086e25:.1f} Gpc")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    title_bits = []
    if gbm:
        title_bits.append(f"GBM β₉₅={gbm['beta_95']:.1e}")
    if lle:
        title_bits.append(f"LLE β₉₅={lle['beta_95']:.1e}")
    if ext:
        title_bits.append(f"Ext β₉₅={ext['beta_95']:.1e}")
    title_extra = (" | " + ", ".join(title_bits)) if title_bits else ""
    fig.suptitle(
        f"CH dispersion: first-principles + Fermi GRB 090510{title_extra}",
        fontsize=12, y=1.01,
    )
    out = OUTPUT / "ch_dispersion_fermi_combined.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def print_report(
    ch: CHParams,
    gbm: dict[str, float] | None,
    lle: dict[str, float] | None,
    ext: dict[str, float] | None,
) -> None:
    beta_lim = beta_bound_from_timing(SIGMA_T_FERMI_S, D_L_GRB_M, E_PHOTON_GEV[0], E_PHOTON_GEV[-1])
    void_beta = float(ch.beta_max * chi(0.0, ch.grad_rho_crit))

    print("=" * 76)
    print("CH DISPERSION — COMBINED FIRST-PRINCIPLES + FERMI GRB 090510")
    print("=" * 76)
    print(f"\nCH (ξ = {ch.xi:.3e} m):  β_max = {ch.beta_max:.3e}  |  β_eff(void) = {void_beta:.3e}")
    print(f"Fermi LAT bound:        β ≲ {BETA_FERMI_BOUND:.3e}  (E_QG,2 ~ {E_QG_FERMI_GEV:.0e} GeV)")
    print(f"GRB timing bound:       β ≲ {beta_lim:.3e}  (σ_t={SIGMA_T_FERMI_S}s)")
    print(f"\nNaive always-on CH:     {verdict_always_on(ch.beta_max)}")
    print(f"Gradient-gated void:    {verdict_gated(void_beta)} — null GRB is correct CH prediction")

    if gbm:
        print(f"\nReal Fermi GBM (MeV):")
        print(f"  β_95 = {gbm['beta_95']:.3e}  (p={gbm.get('p_value', float('nan')):.2f})")
    if lle:
        print(f"\nReal Fermi LAT LLE (GeV):")
        print(f"  β_95 = {lle['beta_95']:.3e}  (p={lle.get('p_value', float('nan')):.2f}, n={int(lle.get('n_photons',0))})")
    if ext:
        print(f"\nReal Fermi LAT extended (TRANSIENT, incl. ~30 GeV):")
        print(f"  β_95 = {ext['beta_95']:.3e}  (p={ext.get('p_value', float('nan')):.2f}, n={int(ext.get('n_photons',0))})")
        print(f"  E_QG,2 ≳ {ext.get('e_qg_gev', 0):.2e} GeV")
    if gbm or lle or ext:
        print("  → Null fits consistent with gradient-gated CH")
    else:
        print(f"\nNo real Fermi cache.")
        print(f"  Run: python {FERMI_GBM_SCRIPT.name}")
        print(f"       python {FERMI_LLE_SCRIPT.name}")
        print(f"       python {FERMI_EXT_SCRIPT.name}")

    print(f"\nPrimary falsifiable test: photon Δt vs |∇ρ| lab knob (Prediction #7), not void GRBs.")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="CH dispersion + Fermi GRB combined test")
    parser.add_argument("--xi", type=float, default=1e-3, help="Healing length ξ [m]")
    parser.add_argument("--run-fermi", action="store_true", help="Run Fermi download/fit first")
    args = parser.parse_args()

    if args.run_fermi:
        gbm, lle, ext = run_fermi_pipeline()
    else:
        gbm = load_fermi_results()
        lle = load_lle_results()
        ext = load_extended_results()
    ch = CHParams(xi=args.xi)
    print_report(ch, gbm, lle, ext)
    out = plot_combined(ch, gbm, lle, ext)
    print(f"\nSaved {out}")

    summary = OUTPUT / "ch_dispersion_fermi_combined.txt"
    with summary.open("w") as f:
        f.write(f"xi_m={ch.xi}\n")
        f.write(f"beta_max={ch.beta_max}\n")
        f.write(f"beta_eff_void={float(ch.beta_max * chi(0.0, ch.grad_rho_crit))}\n")
        f.write(f"beta_fermi_lat={BETA_FERMI_BOUND}\n")
        f.write(f"verdict_always_on={verdict_always_on(ch.beta_max)}\n")
        f.write(f"verdict_gated_void={verdict_gated(float(ch.beta_max * chi(0.0, ch.grad_rho_crit)))}\n")
        for tag, data in [("gbm", gbm), ("lle", lle), ("ext", ext)]:
            if data:
                for k in ("beta_95", "beta", "e_qg_gev", "n_photons", "p_value"):
                    if k in data:
                        f.write(f"{tag}_{k}={data[k]}\n")
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
