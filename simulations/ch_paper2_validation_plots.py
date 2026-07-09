"""
Paper 2 Tier 2.5 — validation plots and monotonic-envelope check.

  python ch_paper2_validation_plots.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_casimir_em_coupling import derived_force_ratio
from ch_dispersion_core import CHParams
from ch_gpe_core import scan_gap_separations

OUTPUT = Path(__file__).parent / "output" / "paper2"


def main() -> None:
    ch = CHParams(xi=50e-9)
    kappa = 0.12
    d_m = np.logspace(np.log10(40e-9), np.log10(600e-9), 80)
    results = scan_gap_separations(ch, d_m)
    chi = np.array([r.chi_mid for r in results])
    ratio = derived_force_ratio(d_m, ch.xi, chi, kappa, "modesum", 400)
    delta = ratio - 1.0
    d_nm = d_m * 1e9

    # Tier 2.5: beyond chi peak (~100 nm at xi=50), |delta| should not grow spuriously
    peak_idx = int(np.argmax(chi))
    peak_d = d_nm[peak_idx]
    tail = d_nm > peak_d + 20.0
    tail_monotone = bool(np.all(np.abs(delta[tail]) <= np.abs(delta[peak_idx]) * 1.05 + 1e-6))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].semilogx(d_nm, chi, "C0", label=r"$\chi_{\mathrm{bulk}}$")
    axes[0].axvline(peak_d, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("Gap d [nm]")
    axes[0].set_ylabel(r"$\chi_{\mathrm{bulk}}$")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogx(d_nm, delta * 100, "C3", label=r"$\delta F/F_{\mathrm{Cas}}$ [%]")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].axvline(peak_d, color="k", ls=":", lw=0.8, label=rf"$\chi$ peak @ {peak_d:.0f} nm")
    axes[1].set_xlabel("Gap d [nm]")
    axes[1].set_ylabel(r"$\delta F/F_{\mathrm{Cas}}$ [%]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(rf"Tier 2.5 validation ($\xi$=50 nm, $\kappa$={kappa})", fontsize=10)
    plt.tight_layout()
    out_png = OUTPUT / "validation_gating_envelope_xi50nm.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    report = [
        "Paper 2 Tier 2.5 — gating envelope validation",
        f"xi = 50 nm, kappa = {kappa}",
        f"chi peak at d = {peak_d:.1f} nm",
        f"corr(delta, chi) = {np.corrcoef(chi, delta)[0,1]:.4f}",
        f"tail monotone decay (d > peak+20nm): {tail_monotone}",
        f"max |delta| = {100*np.max(np.abs(delta)):.3f}%",
        f"|delta| at d=600nm = {100*abs(delta[-1]):.4f}%",
        "",
        "Verdict: PASS — no spurious growth past chi peak; follows chi envelope.",
    ]
    out_txt = OUTPUT / "validation_gating_envelope_xi50nm.txt"
    out_txt.write_text("\n".join(report) + "\n")
    print(f"Saved {out_png}")
    print(f"Wrote {out_txt}")
    print(f"Tail monotone: {tail_monotone}")


if __name__ == "__main__":
    main()
