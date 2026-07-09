"""
Paper 2 Tier 1.3 — sweep healing length ξ and summarize a_vac*(ξ), α_max(ξ), grain κ.

  python ch_paper2_xi_sweep.py
  python ch_paper2_xi_sweep.py --xi-list 10e-9 20e-9 50e-9 100e-9 --k-heal 2.0
  python ch_paper2_xi_sweep.py --quick
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

from ch_casimir_em_coupling import derived_force_ratio
from ch_dispersion_core import CHParams
from ch_gpe_core import scan_gap_separations
from ch_paper2_kappa_derive import predict_alpha_max
from ch_paper2_grain_polarizability import (
    KAPPA_BENCHMARK,
    alpha_max_from_kappa_linear,
    preferred_kappa_band,
)
from ch_paper2_supersolid_ground_state import (
    minimize_ground_state,
    minimize_ground_state_staged,
    representative_d_hat,
)

OUTPUT = Path(__file__).parent / "output" / "paper2"
DATA = Path(__file__).parent / "data" / "paper2"

DEFAULT_XI = [10e-9, 20e-9, 50e-9, 100e-9]
KAPPA_BENCH = 0.12


@dataclass
class XiSweepRow:
    xi_m: float
    xi_nm: float
    d_hat: float
    d_rep_nm: float
    eta_star: float
    a_vac_star_nm: float
    a_ratio_2pi_xi: float
    a_ratio_emod_only: float
    kappa_grain_cl: float
    kappa_grain_fs: float
    alpha_bench: float
    alpha_grain_cl: float
    alpha_grain_fs: float
    corr_delta_chi: float
    peak_delta_pct: float


def gating_stats(ch: CHParams, kappa: float, n_d: int = 30, n_max: int = 400) -> tuple[float, float]:
    d_m = np.logspace(np.log10(40e-9), np.log10(600e-9), n_d)
    results = scan_gap_separations(ch, d_m)
    chi = np.array([r.chi_mid for r in results])
    ratio = derived_force_ratio(d_m, ch.xi, chi, kappa, "modesum", n_max)
    delta = ratio - 1.0
    corr = float(np.corrcoef(chi, delta)[0, 1]) if np.std(chi) > 0 else float("nan")
    peak = float(np.max(np.abs(delta)))
    return corr, peak


def row_for_xi(
    xi_m: float,
    k_heal: float,
    kappa_bench: float,
    quick: bool,
    n_max: int,
) -> XiSweepRow:
    ch = CHParams(xi=xi_m)
    d_hat, d_rep = representative_d_hat(ch)

    gs_emod = minimize_ground_state(d_hat, ch.xi, k_heal=0.0)
    gs = minimize_ground_state_staged(d_hat, ch.xi, k_heal=k_heal) if k_heal > 0 else gs_emod

    n_use = 300 if quick else n_max
    pred = predict_alpha_max(ch, gs, kappa_bench, n_max=n_use)
    alpha_bench = pred["alpha_max_pred"]

    band = preferred_kappa_band(ch, gs.eta_star)
    corr, peak = gating_stats(ch, kappa_bench, n_max=min(n_use, 400))

    return XiSweepRow(
        xi_m=xi_m,
        xi_nm=xi_m * 1e9,
        d_hat=d_hat,
        d_rep_nm=d_rep * 1e9,
        eta_star=gs.eta_star,
        a_vac_star_nm=gs.a_vac_star_m * 1e9,
        a_ratio_2pi_xi=gs.ratio_to_hyp,
        a_ratio_emod_only=gs_emod.ratio_to_hyp,
        kappa_grain_cl=band.kappa_mid,
        kappa_grain_fs=band.kappa_high,
        alpha_bench=alpha_bench,
        alpha_grain_cl=alpha_max_from_kappa_linear(band.kappa_mid, alpha_bench, KAPPA_BENCHMARK),
        alpha_grain_fs=alpha_max_from_kappa_linear(band.kappa_high, alpha_bench, KAPPA_BENCHMARK),
        corr_delta_chi=corr,
        peak_delta_pct=100.0 * peak,
    )


def write_csv(path: Path, rows: list[XiSweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "xi_nm",
        "d_hat",
        "d_rep_nm",
        "eta_star",
        "a_vac_star_nm",
        "a_ratio_2pi_xi",
        "a_ratio_emod_only",
        "kappa_grain_cl",
        "kappa_grain_fs",
        "alpha_bench_kappa0p12",
        "alpha_grain_cl_pct",
        "alpha_grain_fs_pct",
        "corr_delta_chi",
        "peak_delta_pct",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow(
                [
                    f"{r.xi_nm:.1f}",
                    f"{r.d_hat:.4f}",
                    f"{r.d_rep_nm:.2f}",
                    f"{r.eta_star:.4f}",
                    f"{r.a_vac_star_nm:.2f}",
                    f"{r.a_ratio_2pi_xi:.4f}",
                    f"{r.a_ratio_emod_only:.4f}",
                    f"{r.kappa_grain_cl:.6e}",
                    f"{r.kappa_grain_fs:.6e}",
                    f"{r.alpha_bench:.6e}",
                    f"{100*r.alpha_grain_cl:.4f}",
                    f"{100*r.alpha_grain_fs:.4f}",
                    f"{r.corr_delta_chi:.4f}",
                    f"{r.peak_delta_pct:.4f}",
                ]
            )
    print(f"Wrote {path}")


def plot_summary(rows: list[XiSweepRow], out_png: Path, k_heal: float) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    xi = np.array([r.xi_nm for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    axes[0, 0].plot(xi, [r.a_ratio_2pi_xi for r in rows], "o-", label=rf"staged $K_{{\mathrm{{heal}}}}$={k_heal}")
    axes[0, 0].plot(xi, [r.a_ratio_emod_only for r in rows], "s--", label=r"$E_{\mathrm{mod}}$ only")
    axes[0, 0].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[0, 0].set_xlabel(r"$\xi$ [nm]")
    axes[0, 0].set_ylabel(r"$a^*/(2\pi\xi)$")
    axes[0, 0].set_title("Vacuum lattice period")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(xi, [100 * r.alpha_bench for r in rows], "o-", label=rf"$\kappa$={KAPPA_BENCH}")
    axes[0, 1].semilogy(xi, [100 * r.alpha_grain_cl for r in rows], "s--", label="preferred κ mid")
    axes[0, 1].semilogy(xi, [100 * r.alpha_grain_fs for r in rows], "^--", label="preferred κ high")
    axes[0, 1].set_xlabel(r"$\xi$ [nm]")
    axes[0, 1].set_ylabel(r"$\alpha_{\max}$ [%]")
    axes[0, 1].set_title("Ripple amplitude")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3, which="both")

    axes[1, 0].semilogy(xi, [r.kappa_grain_cl for r in rows], "o-", label=r"$\kappa$ mid (classical)")
    axes[1, 0].semilogy(xi, [r.kappa_grain_fs for r in rows], "s--", label=r"$\kappa$ high (CM cap)")
    axes[1, 0].axhline(KAPPA_BENCH, color="C3", ls=":", label="benchmark")
    axes[1, 0].set_xlabel(r"$\xi$ [nm]")
    axes[1, 0].set_ylabel(r"$\kappa$")
    axes[1, 0].set_title("Grain OOM susceptibility")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3, which="both")

    axes[1, 1].plot(xi, [r.corr_delta_chi for r in rows], "o-", color="C2")
    axes[1, 1].set_xlabel(r"$\xi$ [nm]")
    axes[1, 1].set_ylabel(r"corr($\delta F/F$, $\chi$)")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Gating (κ=0.12)")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Paper 2 Tier 1.3 — ξ sweep summary", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(path: Path, rows: list[XiSweepRow], k_heal: float) -> None:
    lines = [
        "Paper 2 Tier 1.3 — xi sweep summary",
        f"K_heal = {k_heal}",
        f"kappa benchmark = {KAPPA_BENCH}",
        "",
    ]
    for r in rows:
        lines.extend(
            [
                f"xi = {r.xi_nm:.1f} nm",
                f"  d_hat = {r.d_hat:.3f}, d_rep = {r.d_rep_nm:.1f} nm",
                f"  eta* = {r.eta_star:.4f}",
                f"  a*/(2pi xi) = {r.a_ratio_2pi_xi:.4f} (E_mod only: {r.a_ratio_emod_only:.4f})",
                f"  alpha_max @ kappa=0.12 = {100*r.alpha_bench:.3f}%",
                f"  grain alpha: cl {100*r.alpha_grain_cl:.3f}%, fs {100*r.alpha_grain_fs:.3f}%",
                f"  corr(delta, chi) = {r.corr_delta_chi:.4f}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 xi sweep aggregator")
    parser.add_argument("--xi-list", type=float, nargs="+", default=DEFAULT_XI)
    parser.add_argument("--k-heal", type=float, default=2.0)
    parser.add_argument("--n-max", type=int, default=800)
    parser.add_argument("--quick", action="store_true", help="Faster mode-sum cutoff")
    args = parser.parse_args()

    rows: list[XiSweepRow] = []
    for xi in args.xi_list:
        print(f"\n--- xi = {xi*1e9:.0f} nm ---")
        rows.append(
            row_for_xi(xi, args.k_heal, KAPPA_BENCH, quick=args.quick, n_max=args.n_max)
        )
        r = rows[-1]
        print(
            f"  a*/(2πξ)={r.a_ratio_2pi_xi:.3f}, α_bench={100*r.alpha_bench:.2f}%, "
            f"α_grain={100*r.alpha_grain_cl:.3f}%"
        )

    write_csv(OUTPUT / "xi_sweep_summary.csv", rows)
    plot_summary(rows, OUTPUT / "xi_sweep_summary.png", args.k_heal)
    write_report(OUTPUT / "xi_sweep_summary.txt", rows, args.k_heal)


if __name__ == "__main__":
    main()
