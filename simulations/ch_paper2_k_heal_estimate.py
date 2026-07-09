"""
Paper 2 Tier 1.1 — OOM estimate of healing-lock stiffness K_heal.

Healing energy (dimensionless):
  E_heal = (K_heal/2) (2π/a_hat − 1)² ∫ ρ_TF dẑ

Estimates K_heal needed to shift minimum from a_hat ≈ π (E_mod) toward 2π:
  K_heal ~ ΔE_mod / [0.5 * bulk * (mismatch_π² − mismatch_{2π}²)]

Also compares to grain-scale elastic scale and simulated K_heal scan crossing.

  python ch_paper2_k_heal_estimate.py
  python ch_paper2_k_heal_estimate.py --xi 50e-9 --staged
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_paper2_supersolid_energy import (
    A_HYP_FACTOR,
    supersolid_modulation_energy,
    tf_bulk_integral,
)
from ch_paper2_supersolid_ground_state import (
    find_k_for_period,
    minimize_ground_state,
    minimize_ground_state_staged,
    representative_d_hat,
    scan_healing_stiffness,
)

OUTPUT = Path(__file__).parent / "output" / "paper2"


@dataclass
class KHealEstimate:
    k_heal_energy_balance: float
    k_heal_grain_elastic: float
    k_heal_sim_crossing: float | None
    e_mod_at_pi: float
    e_mod_at_2pi: float
    delta_e_mod: float
    bulk_tf: float
    eta_star: float


def estimate_k_heal_energy_balance(d_hat: float, eta: float) -> KHealEstimate:
    """K_heal ~ E_mod cost to move from a_hat=pi to 2pi at fixed eta*."""
    a_pi = np.pi
    a_2pi = A_HYP_FACTOR
    e_pi = supersolid_modulation_energy(d_hat, eta, a_pi)
    e_2pi = supersolid_modulation_energy(d_hat, eta, a_2pi)
    bulk = tf_bulk_integral(d_hat)
    mismatch_pi = (2.0 * np.pi / a_pi) - 1.0
    # E_heal(2pi)=0; E_heal(pi) = 0.5 K bulk mismatch_pi^2. Balance: K * E_heal_unit = E_mod(2pi)-E_mod(pi).
    delta_e = max(e_2pi - e_pi, 0.0)
    heal_unit = 0.5 * bulk * mismatch_pi**2
    k_balance = delta_e / heal_unit if heal_unit > 0 else float("nan")
    return KHealEstimate(
        k_heal_energy_balance=k_balance,
        k_heal_grain_elastic=float("nan"),
        k_heal_sim_crossing=None,
        e_mod_at_pi=e_pi,
        e_mod_at_2pi=e_2pi,
        delta_e_mod=delta_e,
        bulk_tf=bulk,
        eta_star=eta,
    )


def grain_elastic_k_heal(ch: CHParams, d_hat: float) -> float:
    """
    Natural dimensionless stiffness ~ E_mod scale / healing penalty unit.

    Uses E_mod(eta*, pi) as the vacuum modulation energy scale in the cavity.
    """
    eta = 0.2  # typical; overwritten in main from gs0
    bulk = tf_bulk_integral(d_hat)
    mismatch_pi = (2.0 * np.pi / np.pi) - 1.0
    heal_unit = 0.5 * bulk * mismatch_pi**2
    e_pi = supersolid_modulation_energy(d_hat, eta, np.pi)
    return float(e_pi / heal_unit) if heal_unit > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="K_heal OOM estimate")
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--staged", action="store_true", default=True)
    parser.add_argument("--no-staged", action="store_false", dest="staged")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_hat, d_rep = representative_d_hat(ch)
    gs0 = minimize_ground_state(d_hat, ch.xi, k_heal=0.0)
    eta = gs0.eta_star

    est = estimate_k_heal_energy_balance(d_hat, eta)
    est.k_heal_grain_elastic = grain_elastic_k_heal(ch, d_hat)
    # Recompute with actual eta for grain_elastic label (E_mod scale)
    bulk = tf_bulk_integral(d_hat)
    heal_unit = 0.5 * bulk * ((2.0 * np.pi / np.pi) - 1.0) ** 2
    est.k_heal_grain_elastic = supersolid_modulation_energy(d_hat, eta, np.pi) / heal_unit

    scan = scan_healing_stiffness(d_hat, ch.xi, staged=args.staged)
    k_cross = find_k_for_period(scan, target_ratio=1.0)
    est.k_heal_sim_crossing = k_cross

    tag = f"xi{int(args.xi * 1e9)}nm"
    lines = [
        "Paper 2 Tier 1.1 — K_heal order-of-magnitude estimates",
        f"xi = {ch.xi*1e9:.1f} nm",
        f"d_hat = {d_hat:.4f}, d_rep = {d_rep*1e9:.1f} nm",
        f"eta* (E_mod) = {eta:.4f}",
        f"bulk TF integral = {est.bulk_tf:.4f}",
        "",
        "E_mod at a_hat = pi:",
        f"  E_mod(pi) = {est.e_mod_at_pi:.6e}",
        f"  E_mod(2pi) = {est.e_mod_at_2pi:.6e}",
        f"  delta E_mod = {est.delta_e_mod:.6e}",
        "",
        "K_heal estimates (dimensionless stiffness):",
        f"  energy balance (E_mod 2pi-pi) = {est.k_heal_energy_balance:.6e}",
        f"  E_mod scale / heal unit     = {est.k_heal_grain_elastic:.6e}",
        f"  simulated crossing a*/(2pi xi)=1 = {k_cross}",
        "",
        "Ratio sim / energy-balance:",
    ]
    if k_cross and est.k_heal_energy_balance > 0:
        lines.append(f"  K_cross / K_balance = {k_cross / est.k_heal_energy_balance:.3f}")
    if est.k_heal_grain_elastic > 0 and k_cross:
        lines.append(f"  K_cross / K_Emod    = {k_cross / est.k_heal_grain_elastic:.3f}")
    lines.append(f"  K_working (2.0) gives a*/(2pi xi) ~ 0.95 in staged scan")

    lines.extend(
        [
            "",
            "Interpretation:",
            "- If K_balance ~ K_cross, healing lock is fixing E_mod pi-vs-2pi competition",
            "- K_heal = 2 (working value) is O(1-10) on these scales, not derived exactly",
            "- Lab still discriminates pi xi vs 2 pi xi regardless",
        ]
    )
    out_txt = OUTPUT / f"k_heal_estimate_{tag}.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_txt}")

    # Plot scan
    ks = np.array([s[0] for s in scan])
    ratios = np.array([s[1].ratio_to_hyp for s in scan])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(ks, ratios, "o-", ms=4)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    if k_cross:
        ax.axvline(k_cross, color="C3", ls="--", label=rf"$K_{{\mathrm{{cross}}}}$={k_cross:.3g}")
    if est.k_heal_energy_balance > 0:
        ax.axvline(
            est.k_heal_energy_balance,
            color="C1",
            ls="--",
            label=rf"$K_{{\mathrm{{bal}}}}$={est.k_heal_energy_balance:.3g}",
        )
    ax.set_xlabel(r"$K_{\mathrm{heal}}$")
    ax.set_ylabel(r"$a^*/(2\pi\xi)$")
    ax.set_title(rf"Healing lock scan ($\xi$={ch.xi*1e9:.0f} nm, staged={args.staged})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(OUTPUT / f"k_heal_estimate_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT / f'k_heal_estimate_{tag}.png'}")

    print(f"\nK_balance={est.k_heal_energy_balance:.4g}, K_grain={est.k_heal_grain_elastic:.4g}, K_cross={k_cross}")


if __name__ == "__main__":
    main()
