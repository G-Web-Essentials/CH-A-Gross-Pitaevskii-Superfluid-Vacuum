"""
Paper 2 — EM–GPE Casimir coupling (Phase 1 + 1b).

Coupling C1 (docs/paper-2/coupling-spec.md):
  ε(z) = ε₀ [1 + κ χ_bulk(d) (1 − ρ/ρ_in)]

Methods:
  toy      — Phase 1: ∫ δε sin²(πu) du (single-mode kernel)
  modesum  — Phase 1b: perturbative TE mode sum, n²-weighted (default)
  both     — compare toy vs modesum

  python ch_casimir_em_coupling.py
  python ch_casimir_em_coupling.py --method both --n-max 800
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from casimir_ripple_sim import casimir_force, ripple_factor
from ch_casimir_mode_sum import (
    convergence_check,
    mode_sum_force_correction_te_tm,
)
from ch_dispersion_core import CHParams
from ch_gpe_core import scan_gap_separations, thomas_fermi_box_profile

DATA = Path(__file__).parent / "data" / "paper2"
OUTPUT = Path(__file__).parent / "output" / "paper2"
A_PLATE = 1e-6  # m² — same as casimir_ripple_sim.py


def epsilon_relative(
    z_hat: np.ndarray,
    d_hat: float,
    chi_bulk: float,
    kappa: float,
) -> np.ndarray:
    """ε/ε₀ from coupling C1."""
    rho = thomas_fermi_box_profile(z_hat, d_hat)
    return 1.0 + kappa * chi_bulk * (1.0 - rho)


def relative_force_correction_toy(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    n_z: int = 4000,
) -> float:
    """Phase 1 toy: ∫₀¹ δε(u) sin²(πu) du."""
    if chi_bulk <= 0.0 or d_m <= 0.0:
        return 0.0
    d_hat = d_m / xi_m
    if d_hat < 0.04:
        return 0.0
    z_hat = np.linspace(0.0, d_hat, n_z)
    u = z_hat / d_hat
    eps = epsilon_relative(z_hat, d_hat, chi_bulk, kappa)
    delta = eps - 1.0
    weight = np.sin(np.pi * u) ** 2
    return float(np.trapz(delta * weight, u))


def relative_force_correction(
    d_m: float,
    xi_m: float,
    chi_bulk: float,
    kappa: float,
    method: str = "modesum",
    n_max: int = 800,
) -> float:
    if method == "toy":
        return relative_force_correction_toy(d_m, xi_m, chi_bulk, kappa)
    if method == "modesum":
        return mode_sum_force_correction_te_tm(d_m, xi_m, chi_bulk, kappa, n_max=n_max)
    raise ValueError(f"Unknown method: {method}")


def derived_force_ratio(
    d_m: np.ndarray,
    xi_m: float,
    chi_bulk: np.ndarray,
    kappa: float,
    method: str = "modesum",
    n_max: int = 800,
) -> np.ndarray:
    corr = np.array(
        [
            relative_force_correction(float(d), xi_m, float(c), kappa, method, n_max)
            for d, c in zip(d_m, chi_bulk)
        ]
    )
    return 1.0 + corr


def fit_ripple_amplitude(d_m: np.ndarray, ratio: np.ndarray, a_vac: float) -> float:
    """Fit α in 1 + α cos(2πd/a_vac) to derived ratio (phase fixed 0)."""

    def model(d, alpha):
        return ripple_factor(d, alpha, a_vac, 0.0)

    try:
        popt, _ = curve_fit(model, d_m, ratio, p0=[0.05], maxfev=5000)
        return float(popt[0])
    except Exception:
        return float("nan")


def write_derived_csv(
    path: Path,
    d_m: np.ndarray,
    chi_bulk: np.ndarray,
    ratio: np.ndarray,
    ratio_toy: np.ndarray | None,
    f_cas: np.ndarray,
    kappa: float,
    xi_m: float,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "d_m",
            "d_nm",
            "k_per_nm",
            "chi_bulk",
            "F_over_F_Cas",
            "delta_F_over_F_Cas",
            "method",
        ]
        if ratio_toy is not None:
            header.extend(["delta_F_over_F_Cas_toy", "delta_F_over_F_Cas_modesum"])
        header.extend(["F_Cas_N", "kappa", "xi_m"])
        w.writerow(header)
        for i, (d, c, r, fc) in enumerate(zip(d_m, chi_bulk, ratio, f_cas)):
            row = [
                f"{d:.6e}",
                f"{d * 1e9:.6f}",
                f"{1.0 / (d * 1e9):.6e}",
                f"{c:.6e}",
                f"{r:.8e}",
                f"{r - 1.0:.8e}",
                method,
            ]
            if ratio_toy is not None:
                row.extend([f"{ratio_toy[i] - 1.0:.8e}", f"{ratio[i] - 1.0:.8e}"])
            row.extend([f"{fc:.6e}", f"{kappa:.6e}", f"{xi_m:.6e}"])
            w.writerow(row)
    print(f"Wrote {path}")


def plot_results(
    d_nm: np.ndarray,
    chi_bulk: np.ndarray,
    ratio: np.ndarray,
    ratio_toy: np.ndarray | None,
    kappa: float,
    xi_m: float,
    method: str,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ncols = 3 if ratio_toy is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 2:
        axes = np.array(axes)

    axes[0, 0].semilogx(d_nm, chi_bulk, "o-", color="C0")
    axes[0, 0].set_xlabel("Gap d [nm]")
    axes[0, 0].set_ylabel(r"$\chi_{\mathrm{bulk}}$ (GPE)")
    axes[0, 0].set_title("Layer 1 input")
    axes[0, 0].grid(alpha=0.3, which="both")

    delta = ratio - 1.0
    axes[0, 1].semilogx(d_nm, delta, "o-", color="C3", label=method)
    if ratio_toy is not None:
        axes[0, 1].semilogx(d_nm, ratio_toy - 1.0, "s--", color="C1", alpha=0.8, label="toy")
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set_xlabel("Gap d [nm]")
    axes[0, 1].set_ylabel(r"$\delta F/F_{\mathrm{Cas}}$")
    axes[0, 1].set_title(rf"Derived correction ($\kappa$={kappa})")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3, which="both")

    axes[1, 0].plot(chi_bulk, delta, "o", color="C2", label=method)
    if ratio_toy is not None:
        axes[1, 0].plot(chi_bulk, ratio_toy - 1.0, "s", color="C1", alpha=0.6, label="toy")
    axes[1, 0].set_xlabel(r"$\chi_{\mathrm{bulk}}$")
    axes[1, 0].set_ylabel(r"$\delta F/F_{\mathrm{Cas}}$")
    axes[1, 0].set_title("Gating check")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    if len(chi_bulk) > 2 and np.std(chi_bulk) > 0:
        r = float(np.corrcoef(chi_bulk, delta)[0, 1])
        axes[1, 0].text(0.05, 0.92, rf"corr = {r:.3f}", transform=axes[1, 0].transAxes)

    if ratio_toy is not None and ncols == 3:
        rel_diff = np.abs(delta - (ratio_toy - 1.0))
        axes[0, 2].semilogx(d_nm, rel_diff, "o-", color="C5")
        axes[0, 2].set_xlabel("Gap d [nm]")
        axes[0, 2].set_ylabel(r"$|\delta_{\mathrm{ms}} - \delta_{\mathrm{toy}}|$")
        axes[0, 2].set_title("Toy vs mode-sum")
        axes[0, 2].grid(alpha=0.3, which="both")

        axes[1, 1].axis("off")
        axes[1, 2].axis("off")
    else:
        axes[1, 1].axis("off")

    phase_label = "Phase 1b" if method == "modesum" else "Phase 1"
    fig.suptitle(
        rf"Paper 2 {phase_label}: EM coupling C1 ($\xi$={xi_m*1e9:.0f} nm, $\kappa$={kappa})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(
    path: Path,
    ch: CHParams,
    kappa: float,
    d_m: np.ndarray,
    chi_bulk: np.ndarray,
    ratio: np.ndarray,
    ratio_toy: np.ndarray | None,
    method: str,
    n_max: int,
) -> None:
    delta = ratio - 1.0
    corr = float(np.corrcoef(chi_bulk, delta)[0, 1]) if np.std(chi_bulk) > 0 else float("nan")
    peak_delta = float(np.max(np.abs(delta)))
    peak_chi = float(np.max(chi_bulk))

    conv = convergence_check(80e-9, ch.xi, float(np.max(chi_bulk)), kappa)

    with path.open("w") as f:
        f.write(f"Paper 2 — EM–GPE Casimir coupling ({method})\n")
        f.write(f"xi [m] = {ch.xi:.6e}\n")
        f.write(f"kappa = {kappa:.6e}\n")
        f.write(f"n_max (mode sum) = {n_max}\n\n")
        f.write(f"peak chi_bulk = {peak_chi:.6e}\n")
        f.write(f"peak |delta F/F_Cas| = {peak_delta:.6e}\n")
        f.write(f"corr(delta F/F_Cas, chi_bulk) = {corr:.6f}\n\n")

        if ratio_toy is not None:
            dt = ratio_toy - 1.0
            corr_toy = float(np.corrcoef(chi_bulk, dt)[0, 1]) if np.std(chi_bulk) > 0 else float("nan")
            max_diff = float(np.max(np.abs(delta - dt)))
            f.write(f"toy: peak |delta| = {np.max(np.abs(dt)):.6e}, corr = {corr_toy:.6f}\n")
            f.write(f"max |delta_modesum - delta_toy| = {max_diff:.6e}\n\n")

        f.write("Mode-sum cutoff convergence at d=80 nm, peak chi:\n")
        for n, c in conv:
            f.write(f"  n_max={n:5d}  correction={c:.6e}\n")

        f.write("\nInterpretation:\n")
        f.write("- Phase 1b: TE mode sum with n^2 weights (TE+TM factor 2)\n")
        f.write("- chi -> 0 => delta -> 0 (QFT limit)\n")
        f.write("- Cosine period a_vac still Phase 2\n")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 EM–GPE Casimir coupling")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=40e-9, help="Min gap [m]")
    parser.add_argument("--d-max", type=float, default=600e-9, help="Max gap [m]")
    parser.add_argument("--n-d", type=int, default=40, help="Gap points")
    parser.add_argument("--kappa", type=float, default=0.12, help="EM coupling strength κ")
    parser.add_argument(
        "--method",
        choices=("toy", "modesum", "both"),
        default="modesum",
        help="Phase 1 toy vs Phase 1b mode sum",
    )
    parser.add_argument("--n-max", type=int, default=800, help="Mode-sum cutoff N")
    parser.add_argument("--export-gpe", action="store_true", help="Run GPE export first")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_m = np.logspace(np.log10(args.d_min), np.log10(args.d_max), args.n_d)

    if args.export_gpe:
        from ch_paper2_gpe_export import export_profile_csv, export_scan_csv

        results_pre = scan_gap_separations(ch, d_m)
        tag_xi = f"xi{int(args.xi * 1e9)}nm"
        export_scan_csv(ch, results_pre, DATA / f"gpe_scan_{tag_xi}.csv")
        for label, d_fix in [("d40nm", 40e-9), ("d100nm", 100e-9), ("d600nm", 600e-9)]:
            export_profile_csv(ch, d_fix, DATA / f"rho_profile_{tag_xi}_{label}.csv")

    results = scan_gap_separations(ch, d_m)
    chi_bulk = np.array([r.chi_mid for r in results])
    f_cas = casimir_force(d_m)

    ratio_toy = None
    if args.method == "both":
        ratio_toy = derived_force_ratio(d_m, ch.xi, chi_bulk, args.kappa, "toy")
        ratio = derived_force_ratio(d_m, ch.xi, chi_bulk, args.kappa, "modesum", args.n_max)
        method_tag = "both"
    else:
        ratio = derived_force_ratio(d_m, ch.xi, chi_bulk, args.kappa, args.method, args.n_max)
        method_tag = args.method

    tag = f"xi{int(args.xi * 1e9)}nm_kappa{args.kappa:.2f}_{method_tag}".replace(".", "p")
    write_derived_csv(
        OUTPUT / f"em_coupling_{tag}.csv",
        d_m,
        chi_bulk,
        ratio,
        ratio_toy,
        f_cas,
        args.kappa,
        ch.xi,
        method_tag,
    )

    d_nm = d_m * 1e9
    plot_results(
        d_nm,
        chi_bulk,
        ratio,
        ratio_toy,
        args.kappa,
        ch.xi,
        method_tag,
        OUTPUT / f"em_coupling_{tag}.png",
    )
    write_report(
        OUTPUT / f"em_coupling_{tag}.txt",
        ch,
        args.kappa,
        d_m,
        chi_bulk,
        ratio,
        ratio_toy,
        method_tag,
        args.n_max,
    )

    corr = float(np.corrcoef(chi_bulk, ratio - 1.0)[0, 1])
    print(f"\nMethod: {method_tag}, n_max={args.n_max}")
    print(f"Peak |δF/F_Cas| = {np.max(np.abs(ratio - 1.0)):.4e}")
    print(f"corr(δF/F_Cas, χ_bulk) = {corr:.4f}")


if __name__ == "__main__":
    main()
