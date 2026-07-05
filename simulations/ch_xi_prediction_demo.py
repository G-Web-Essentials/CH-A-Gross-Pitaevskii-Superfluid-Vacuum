#!/usr/bin/env python3
"""
CH ξ prediction demo — compare identification mechanisms.

  python ch_xi_prediction_demo.py
  python ch_xi_prediction_demo.py --a-vac 150e-9 --kc 10.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_xi_prediction import (
    compare_mechanisms,
    consistency_report,
    gpe_turnon_gap_estimate,
    predict_xi_from_casimir_lattice,
    predict_xi_from_kc,
)

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CH ξ prediction mechanisms")
    parser.add_argument("--a-vac", type=float, default=150e-9, help="Ripple period [m]")
    parser.add_argument("--kc", type=float, default=None, help="Threshold k_c [1/nm]")
    parser.add_argument("--e-xi-gev", type=float, default=None, help="E_ξ probe [GeV]")
    args = parser.parse_args()

    print("=" * 72)
    print("CH ξ PREDICTION — proposed identification mechanisms")
    print("=" * 72)

    reports = compare_mechanisms(
        a_vac_m=args.a_vac,
        k_c_per_nm=args.kc,
        e_xi_gev=args.e_xi_gev,
    )
    for rep in reports:
        print()
        for line in rep.summary_lines():
            print(f"  {line}")

    # GPE turn-on estimate for Casimir-predicted ξ
    cas = predict_xi_from_casimir_lattice(args.a_vac)
    ch_cas = cas.ch
    d_turn = gpe_turnon_gap_estimate(ch_cas, target_chi=0.5)
    print("\n--- GPE cross-check (Casimir-predicted ξ) ---")
    print(f"  ξ from a_vac/(2π) = {cas.xi_m:.3e} m")
    if d_turn is not None:
        print(f"  GPE χ_wall ≈ 0.5 at d ≈ {d_turn*1e9:.1f} nm")
        ratio = d_turn / cas.xi_m
        print(f"  d_turn / ξ = {ratio:.3f}  (≈1 if gap-turnon ≡ lattice mechanism)")
    else:
        print("  No χ_wall crossing in 40–600 nm for this ξ")

    # Illustrative: k_c from GPE demo scale (~10 /nm for ξ=50nm turn-on region)
    if args.kc is None:
        demo_kc = 1e9 / (50e-9)  # 20 /nm for ξ = 50 nm
        rep_demo = consistency_report(predict_xi_from_kc(demo_kc))
        print(f"\n--- Illustrative gap turn-on (k_c = {demo_kc:.1f} /nm → ξ = 50 nm) ---")
        for line in rep_demo.summary_lines():
            print(f"  {line}")

    # Plot β_max vs ξ with mechanism markers
    xi_arr = np.logspace(-10, -3, 200)
    beta = 1.5 * xi_arr**2 / (1.054571817e-34) ** 2
    from ch_dispersion_core import BETA_FERMI_BOUND, xi_crit_always_on

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(xi_arr * 1e9, beta, "b-", lw=2, label=r"$\beta_{\max}=(3/2)\xi^2/\hbar^2$")
    ax.axhline(BETA_FERMI_BOUND, color="r", ls="--", label="Fermi bound (always-on)")
    ax.axvline(xi_crit_always_on() * 1e9, color="gray", ls=":", label=r"$\xi_{\mathrm{crit}}$")

    for rep in reports:
        xi_nm = rep.prediction.xi_m * 1e9
        ax.axvline(xi_nm, ls="-.", alpha=0.8, label=f"{rep.prediction.mechanism.value}: {xi_nm:.1f} nm")

    ax.set_xlabel(r"$\xi$ [nm]")
    ax.set_ylabel(r"$\beta_{\max}$ [s/GeV²]")
    ax.set_title("ξ predictions vs always-on Fermi exclusion")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_png = OUTPUT / "ch_xi_prediction.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_txt = OUTPUT / "ch_xi_prediction_report.txt"
    with out_txt.open("w") as f:
        for rep in reports:
            f.write(f"=== {rep.prediction.mechanism.value} ===\n")
            for line in rep.summary_lines():
                f.write(line + "\n")
            f.write("\n")

    print(f"\nWrote {out_png}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
