"""
End-to-end demo: GPE → predicted O(k) → CSV → gradient_threshold_analysis.

1. Scan Casimir gap d; compute χ from 1D GPE (midpoint + wall coupling).
2. Knob k = 1/d_nm so turn-on at small gap ↔ high k (CH threshold shape).
3. Write CSVs with noise; run flat vs threshold fits (joint).

  python ch_gpe_to_threshold_demo.py
  python ch_gpe_to_threshold_demo.py --xi 50e-9 --d-min 40e-9 --d-max 600e-9
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_core import (
    predict_alpha_eff,
    predict_visibility_dip,
    scan_gap_separations,
)

OUTPUT = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data" / "gradient_threshold" / "gpe_demo"
OUTPUT.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def gpe_observables_vs_gap(
    ch: CHParams,
    d_m_array: np.ndarray,
    alpha_max: float,
    vis_max: float,
    coupling: str,
) -> dict[str, np.ndarray]:
    """
    Map gap d to observables via GPE χ values.

    coupling:
      'mid'  — bulk χ (central-half max |∇ρ|); signal channel for gap-d scan
      'wall' — wall χ (flat vs d for d ≳ 2ξ); control forecast on gap scan
      'mixed' — α from χ_wall, ΔV from χ_bulk (cross-channel test)
    """
    results = scan_gap_separations(ch, d_m_array)
    chi_mid = np.array([r.chi_mid for r in results])
    chi_wall = np.array([r.chi_wall for r in results])

    if coupling == "mid":
        alpha = np.array([predict_alpha_eff(c, alpha_max) for c in chi_mid])
        vis = np.array([predict_visibility_dip(c, vis_max) for c in chi_mid])
    elif coupling == "wall":
        alpha = np.array([predict_alpha_eff(c, alpha_max) for c in chi_wall])
        vis = np.array([predict_visibility_dip(c, vis_max) for c in chi_wall])
    else:
        alpha = np.array([predict_alpha_eff(c, alpha_max) for c in chi_wall])
        vis = np.array([predict_visibility_dip(c, vis_max) for c in chi_mid])

    k_inv_nm = 1e9 / d_m_array  # knob: inverse gap [1/nm]
    return {
        "d_m": d_m_array,
        "d_nm": d_m_array * 1e9,
        "k": k_inv_nm,
        "chi_mid": chi_mid,
        "chi_wall": chi_wall,
        "alpha": alpha,
        "vis": vis,
        "results": results,
    }


def write_observable_csv(path: Path, k: np.ndarray, o: np.ndarray, sigma: float) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "O", "sigma"])
        for ki, oi in zip(k, o):
            w.writerow([f"{ki:.8f}", f"{oi:.8f}", f"{sigma:.8f}"])


def add_noise(
    rng: np.random.Generator, signal: np.ndarray, sigma_frac: float, floor: float
) -> tuple[np.ndarray, float]:
    sigma = max(float(np.median(np.abs(signal)) * sigma_frac), floor)
    return signal + rng.normal(0.0, sigma, size=signal.shape), sigma


def plot_gpe_pipeline(data: dict, noisy: dict, out: Path, ch: CHParams) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    k = data["k"]
    axes[0, 0].plot(k, data["chi_mid"], "o-", label=r"$\chi_{\rm mid}$ (GPE)")
    axes[0, 0].plot(k, data["chi_wall"], "s--", label=r"$\chi_{\rm wall}$ (GPE)")
    axes[0, 0].set_xlabel(r"Knob $k = 1/d_{\rm nm}$ [1/nm]")
    axes[0, 0].set_ylabel(r"$\chi$")
    axes[0, 0].set_title("GPE gradient gate vs knob")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(k, data["alpha"], "o-", label=r"$\alpha_{\rm eff}$ (true)")
    axes[0, 1].errorbar(k, noisy["alpha"], yerr=noisy["sigma_alpha"], fmt="s", ms=5, label="with noise")
    axes[0, 1].set_xlabel(r"$k$ [1/nm]")
    axes[0, 1].set_ylabel(r"Ripple $\alpha$")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(k, data["vis"], "o-", label=r"$\Delta V$ (true)")
    axes[1, 0].errorbar(k, noisy["vis"], yerr=noisy["sigma_vis"], fmt="s", ms=5, label="with noise")
    axes[1, 0].set_xlabel(r"$k$ [1/nm]")
    axes[1, 0].set_ylabel(r"Visibility dip")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    d_nm = data["d_nm"]
    axes[1, 1].semilogx(d_nm, data["chi_mid"], "o-", label=r"$\chi_{\rm mid}$")
    axes[1, 1].semilogx(d_nm, data["chi_wall"], "s--", label=r"$\chi_{\rm wall}$")
    axes[1, 1].set_xlabel("Gap d [nm]")
    axes[1, 1].set_ylabel(r"$\chi$")
    axes[1, 1].set_title(rf"vs gap ($\xi$={ch.xi*1e9:.1f} nm)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("GPE → CSV → threshold analysis pipeline", fontsize=11)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def run_demo(
    xi: float,
    d_min: float,
    d_max: float,
    n_d: int,
    alpha_max: float,
    vis_max: float,
    coupling: str,
    noise_frac: float,
    seed: int,
    run_fitter: bool,
) -> None:
    ch = CHParams(xi=xi)
    d_array = np.logspace(np.log10(d_min), np.log10(d_max), n_d)
    data = gpe_observables_vs_gap(ch, d_array, alpha_max, vis_max, coupling)

    rng = np.random.default_rng(seed)
    alpha_noisy, sig_a = add_noise(rng, data["alpha"], noise_frac, floor=0.002)
    vis_noisy, sig_v = add_noise(rng, data["vis"], noise_frac, floor=0.002)
    noisy = {
        "alpha": alpha_noisy,
        "vis": vis_noisy,
        "sigma_alpha": sig_a,
        "sigma_vis": sig_v,
    }

    alpha_csv = DATA / "gpe_alpha_vs_k.csv"
    vis_csv = DATA / "gpe_vis_vs_k.csv"
    write_observable_csv(alpha_csv, data["k"], alpha_noisy, sig_a)
    write_observable_csv(vis_csv, data["k"], vis_noisy, sig_v)
    print(f"Wrote {alpha_csv}")
    print(f"Wrote {vis_csv}")

    plot_gpe_pipeline(data, noisy, OUTPUT / "ch_gpe_to_threshold_demo.png", ch)

    # Summary
    print("\n" + "=" * 60)
    print("GPE → threshold pipeline demo")
    print("=" * 60)
    print(f"ξ = {xi*1e9:.2f} nm, d = {d_min*1e9:.0f}–{d_max*1e9:.0f} nm, coupling={coupling}")
    print(f"χ_mid range: {data['chi_mid'].min():.3f} – {data['chi_mid'].max():.3f}")
    print(f"χ_wall range: {data['chi_wall'].min():.3f} – {data['chi_wall'].max():.3f}")
    print(f"α range: {data['alpha'].min():.4f} – {data['alpha'].max():.4f}")

    if run_fitter:
        print("\n--- Running gradient_threshold_analysis.py ---\n")
        script = Path(__file__).parent / "gradient_threshold_analysis.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--csv-alpha",
                str(alpha_csv),
                "--csv-vis",
                str(vis_csv),
                "--joint",
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="GPE → CSV → threshold analysis demo")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m] (default 50 nm)")
    parser.add_argument("--d-min", type=float, default=40e-9, help="Min gap [m]")
    parser.add_argument("--d-max", type=float, default=600e-9, help="Max gap [m]")
    parser.add_argument("--n-d", type=int, default=14, help="Gap scan points")
    parser.add_argument("--alpha-max", type=float, default=0.12)
    parser.add_argument("--vis-max", type=float, default=0.15)
    parser.add_argument(
        "--coupling",
        choices=["mid", "wall", "mixed"],
        default="mid",
        help="How GPE χ maps to α and ΔV",
    )
    parser.add_argument("--noise", type=float, default=0.08, help="Noise fraction of signal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-fitter", action="store_true", help="Skip threshold analysis step")
    args = parser.parse_args()

    run_demo(
        xi=args.xi,
        d_min=args.d_min,
        d_max=args.d_max,
        n_d=args.n_d,
        alpha_max=args.alpha_max,
        vis_max=args.vis_max,
        coupling=args.coupling,
        noise_frac=args.noise,
        seed=args.seed,
        run_fitter=not args.no_fitter,
    )


if __name__ == "__main__":
    main()
