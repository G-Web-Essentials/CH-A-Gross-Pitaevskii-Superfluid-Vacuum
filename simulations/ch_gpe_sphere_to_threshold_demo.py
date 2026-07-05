"""
End-to-end demo: 2D sphere–plate GPE → O(k) → CSV → gradient_threshold_analysis.

1. Scan sphere radius R at fixed d_min; compute χ_radial from curvature-enhanced |∂ρ/∂r|.
2. Knob k = 1/R_µm (smaller R → tighter curvature → higher k → higher χ).
3. Write CSVs with noise; run flat vs threshold fits (joint).

  python ch_gpe_sphere_to_threshold_demo.py
  python ch_gpe_sphere_to_threshold_demo.py --r-min 100e-9 --r-max 10e-6
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
    scan_sphere_plate_R,
)

OUTPUT = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data" / "gradient_threshold" / "sphere_demo"
OUTPUT.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def gpe_observables_vs_R(
    ch: CHParams,
    d_min_m: float,
    R_m_array: np.ndarray,
    alpha_max: float,
    vis_max: float,
    coupling: str,
) -> dict[str, np.ndarray]:
    """
    Map sphere radius R to observables via 2D GPE χ values.

    coupling:
      'radial' — both channels use χ_radial (curvature-sensitive; varies with R)
      'wall'   — both use χ_wall at r=0 (flat vs R; QFT-like null)
      'mixed'  — α from χ_radial, ΔV from χ_wall (cross-channel)
    """
    results = scan_sphere_plate_R(ch, d_min_m, R_m_array)
    chi_rad = np.array([r.chi_radial for r in results])
    chi_wall = np.array([r.chi_wall for r in results])
    chi_mid = np.array([r.chi_mid for r in results])
    grad_rad = np.array([r.grad_ratio_radial for r in results])

    if coupling == "radial":
        alpha = np.array([predict_alpha_eff(c, alpha_max) for c in chi_rad])
        vis = np.array([predict_visibility_dip(c, vis_max) for c in chi_rad])
    elif coupling == "wall":
        alpha = np.array([predict_alpha_eff(c, alpha_max) for c in chi_wall])
        vis = np.array([predict_visibility_dip(c, vis_max) for c in chi_wall])
    else:
        alpha = np.array([predict_alpha_eff(c, alpha_max) for c in chi_rad])
        vis = np.array([predict_visibility_dip(c, vis_max) for c in chi_wall])

    k_inv_um = 1e6 / R_m_array  # knob: inverse radius [1/µm]
    return {
        "R_m": R_m_array,
        "R_um": R_m_array * 1e6,
        "k": k_inv_um,
        "chi_radial": chi_rad,
        "chi_wall": chi_wall,
        "chi_mid": chi_mid,
        "grad_radial": grad_rad,
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
    span = float(np.max(signal) - np.min(signal))
    sigma = max(span * sigma_frac, floor)
    return signal + rng.normal(0.0, sigma, size=signal.shape), sigma


def plot_pipeline(data: dict, noisy: dict, out: Path, ch: CHParams, d_min_m: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    k = data["k"]

    axes[0, 0].semilogx(k, data["chi_radial"], "o-", label=r"$\chi_{\rm radial}$ (GPE)")
    axes[0, 0].semilogx(k, data["chi_wall"], "s--", label=r"$\chi_{\rm wall}$ (flat)")
    axes[0, 0].set_xlabel(r"Knob $k = 1/R_{\rm \mu m}$ [1/µm]")
    axes[0, 0].set_ylabel(r"$\chi$")
    axes[0, 0].set_title("2D sphere–plate: gradient gate vs curvature knob")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogx(k, data["alpha"], "o-", label=r"$\alpha_{\rm eff}$ (true)")
    axes[0, 1].errorbar(k, noisy["alpha"], yerr=noisy["sigma_alpha"], fmt="s", ms=5, label="with noise")
    axes[0, 1].set_xlabel(r"$k$ [1/µm]")
    axes[0, 1].set_ylabel(r"Ripple $\alpha$")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].semilogx(k, data["vis"], "o-", label=r"$\Delta V$ (true)")
    axes[1, 0].errorbar(k, noisy["vis"], yerr=noisy["sigma_vis"], fmt="s", ms=5, label="with noise")
    axes[1, 0].set_xlabel(r"$k$ [1/µm]")
    axes[1, 0].set_ylabel(r"Visibility dip")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    R_um = data["R_um"]
    axes[1, 1].loglog(R_um, data["grad_radial"], "o-", label=r"$|\partial\rho/\partial r|/|\nabla\rho|_c$")
    axes[1, 1].axhline(1.0, color="k", ls=":", lw=0.8, label="threshold")
    axes[1, 1].set_xlabel("Sphere radius R [µm]")
    axes[1, 1].set_ylabel("Radial gradient ratio")
    axes[1, 1].set_title(rf"$d_{{\min}}$={d_min_m*1e9:.0f} nm, $\xi$={ch.xi*1e9:.0f} nm")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3, which="both")

    fig.suptitle("Sphere–plate GPE → CSV → threshold analysis", fontsize=11)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def run_demo(
    xi: float,
    d_min: float,
    r_min: float,
    r_max: float,
    n_r: int,
    alpha_max: float,
    vis_max: float,
    coupling: str,
    noise_frac: float,
    seed: int,
    run_fitter: bool,
) -> None:
    ch = CHParams(xi=xi)
    R_array = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    data = gpe_observables_vs_R(ch, d_min, R_array, alpha_max, vis_max, coupling)

    rng = np.random.default_rng(seed)
    alpha_noisy, sig_a = add_noise(rng, data["alpha"], noise_frac, floor=1e-5)
    vis_noisy, sig_v = add_noise(rng, data["vis"], noise_frac, floor=1e-5)
    noisy = {
        "alpha": alpha_noisy,
        "vis": vis_noisy,
        "sigma_alpha": sig_a,
        "sigma_vis": sig_v,
    }

    alpha_csv = DATA / "sphere_alpha_vs_k.csv"
    vis_csv = DATA / "sphere_vis_vs_k.csv"
    write_observable_csv(alpha_csv, data["k"], alpha_noisy, sig_a)
    write_observable_csv(vis_csv, data["k"], vis_noisy, sig_v)
    print(f"Wrote {alpha_csv}")
    print(f"Wrote {vis_csv}")

    plot_pipeline(data, noisy, OUTPUT / "ch_gpe_sphere_to_threshold_demo.png", ch, d_min)

    print("\n" + "=" * 60)
    print("Sphere–plate GPE → threshold pipeline demo")
    print("=" * 60)
    print(f"ξ = {xi*1e9:.1f} nm, d_min = {d_min*1e9:.0f} nm, coupling={coupling}")
    print(f"R = {r_min*1e6:.3f}–{r_max*1e6:.1f} µm")
    print(f"χ_radial range: {data['chi_radial'].min():.4f} – {data['chi_radial'].max():.4f}")
    print(f"χ_wall range: {data['chi_wall'].min():.4f} – {data['chi_wall'].max():.4f}")
    print(f"α range: {data['alpha'].min():.6f} – {data['alpha'].max():.4f}")

    if data["chi_radial"].max() < 0.01:
        print("\n⚠ χ_radial very small — try smaller --r-min or smaller --xi for stronger curvature signal.")

    if run_fitter:
        print("\n--- Running gradient_threshold_analysis.py ---\n")
        script = Path(__file__).parent / "gradient_threshold_analysis.py"
        out_png = OUTPUT / "gradient_threshold_analysis_sphere.png"
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
        # Rename output so gap and sphere demos don't overwrite each other
        default_png = OUTPUT / "gradient_threshold_analysis.png"
        if default_png.exists():
            default_png.rename(out_png)
            print(f"Saved {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sphere–plate GPE → CSV → threshold demo")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=100e-9, help="Min gap at contact [m]")
    parser.add_argument("--r-min", type=float, default=50e-9, help="Min sphere radius [m]")
    parser.add_argument("--r-max", type=float, default=5e-6, help="Max sphere radius [m]")
    parser.add_argument("--n-r", type=int, default=14, help="Radius scan points")
    parser.add_argument("--alpha-max", type=float, default=0.12)
    parser.add_argument("--vis-max", type=float, default=0.15)
    parser.add_argument(
        "--coupling",
        choices=["radial", "wall", "mixed"],
        default="radial",
    )
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-fitter", action="store_true")
    args = parser.parse_args()

    run_demo(
        xi=args.xi,
        d_min=args.d_min,
        r_min=args.r_min,
        r_max=args.r_max,
        n_r=args.n_r,
        alpha_max=args.alpha_max,
        vis_max=args.vis_max,
        coupling=args.coupling,
        noise_frac=args.noise,
        seed=args.seed,
        run_fitter=not args.no_fitter,
    )


if __name__ == "__main__":
    main()
