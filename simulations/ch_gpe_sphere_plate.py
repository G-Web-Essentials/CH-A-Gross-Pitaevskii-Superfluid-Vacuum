"""
2D sphere–plate GPE — curvature knob R at fixed minimum gap d_min.

Thomas–Fermi ansatz on axisymmetric (r,z) domain; compares wall, midpoint,
and rim (curvature-enhanced) |∇ρ|/|∇ρ|_c vs sphere radius R.

  python ch_gpe_sphere_plate.py
  python ch_gpe_sphere_plate.py --d-min 100e-9 --r-min 1e-6 --r-max 1e-2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_core import (
    GPE2DSpherePlateResult,
    predict_alpha_eff,
    scan_sphere_plate_R,
    solve_casimir_gap,
)

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def plot_scan(
    ch: CHParams,
    results: list[GPE2DSpherePlateResult],
    d_min_m: float,
    alpha_max: float,
    out_png: Path,
) -> None:
    R_um = np.array([r.R_m for r in results]) * 1e6
    ratio_wall = np.array([r.grad_ratio_wall for r in results])
    ratio_mid = np.array([r.grad_ratio_mid for r in results])
    ratio_rim = np.array([r.grad_ratio_rim for r in results])
    ratio_rad = np.array([r.grad_ratio_radial for r in results])
    chi_rad = np.array([r.chi_radial for r in results])
    alpha_rad = np.array([predict_alpha_eff(c, alpha_max) for c in chi_rad])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].semilogx(R_um, ratio_wall, "o-", label="wall (r=0)")
    axes[0, 0].semilogx(R_um, ratio_mid, "s--", label="mid (r=0)")
    axes[0, 0].semilogx(R_um, ratio_rad, "^-", label=r"$|\partial\rho/\partial r|$ at rim")
    axes[0, 0].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[0, 0].set_xlabel("Sphere radius R [µm]")
    axes[0, 0].set_ylabel(r"$|\nabla\rho|/|\nabla\rho|_c$")
    axes[0, 0].set_title(rf"2D sphere–plate ($d_{{\min}}$={d_min_m*1e9:.0f} nm)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogx(R_um, chi_rad, "o-", color="C2")
    axes[0, 1].set_xlabel("R [µm]")
    axes[0, 1].set_ylabel(r"$\chi_{\rm radial}$")
    axes[0, 1].set_title("Gradient gate (curvature-sensitive)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].semilogx(R_um, alpha_rad, "o-", color="C3")
    axes[1, 0].set_xlabel("R [µm]")
    axes[1, 0].set_ylabel(r"$\alpha_{\rm eff,radial}$")
    axes[1, 0].set_title(rf"Predicted ripple ($\alpha_{{\max}}$={alpha_max})")
    axes[1, 0].grid(alpha=0.3)

    # Curvature enhancement: rim / wall gradient ratio
    enh = ratio_rad / np.maximum(ratio_wall, 1e-30)
    axes[1, 1].semilogx(R_um, enh, "o-")
    axes[1, 1].set_xlabel("R [µm]")
    axes[1, 1].set_ylabel(r"$|\partial\rho/\partial r| / |\partial\rho/\partial z|_{\rm wall}$")
    axes[1, 1].set_title("Curvature enhancement vs flat plate")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        rf"CH 2D sphere–plate GPE ($\xi$={ch.xi:.1e} m)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(
    ch: CHParams,
    results: list[GPE2DSpherePlateResult],
    d_min_m: float,
    alpha_max: float,
    path: Path,
) -> None:
    with path.open("w") as f:
        f.write("CH 2D sphere-plate GPE scan\n")
        f.write(f"xi [m] = {ch.xi:.6e}\n")
        f.write(f"d_min [m] = {d_min_m:.6e}\n\n")
        f.write("R[um]  R/xi   grad_wall  grad_radial  chi_radial  alpha_radial\n")
        for r in results:
            f.write(
                f"{r.R_m*1e6:8.3f} {r.R_hat:8.2e} {r.grad_ratio_wall:10.3e} "
                f"{r.grad_ratio_radial:10.3e} {r.chi_radial:8.3e} "
                f"{predict_alpha_eff(r.chi_radial, alpha_max):10.3e}\n"
            )
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D sphere–plate GPE curvature scan")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=100e-9, help="Min gap at contact [m]")
    parser.add_argument("--r-min", type=float, default=1e-6, help="Min sphere radius [m]")
    parser.add_argument("--r-max", type=float, default=1e-3, help="Max sphere radius [m]")
    parser.add_argument("--n-r", type=int, default=25, help="Radius scan points")
    parser.add_argument("--alpha-max", type=float, default=0.12)
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    R_array = np.logspace(np.log10(args.r_min), np.log10(args.r_max), args.n_r)

    print("=" * 60)
    print("CH 2D sphere–plate GPE — curvature knob R")
    print("=" * 60)
    print(f"ξ = {ch.xi:.3e} m, d_min = {args.d_min*1e9:.1f} nm")
    print(f"Scanning R = {args.r_min*1e6:.3g} – {args.r_max*1e3:.3g} mm ({args.n_r} points)")

    results = scan_sphere_plate_R(ch, args.d_min, R_array)

    r_lo, r_hi = results[0], results[-1]
    print(f"\nSmallest R = {r_lo.R_m*1e6:.3f} µm: radial grad = {r_lo.grad_ratio_radial:.4f}, χ_rad = {r_lo.chi_radial:.4f}")
    print(f"Largest  R = {r_hi.R_m*1e6:.3f} µm: radial grad = {r_hi.grad_ratio_radial:.4e}, χ_rad = {r_hi.chi_radial:.4e}")

    flat_mid = solve_casimir_gap(ch, args.d_min)
    sphere_lo = results[0]
    print(f"\nParallel-plate χ_mid → α = {predict_alpha_eff(flat_mid.chi_mid):.4f}")
    print(f"Sphere R_min radial → α = {predict_alpha_eff(sphere_lo.chi_radial):.4f}")

    if r_lo.grad_ratio_radial > 10 * r_hi.grad_ratio_radial:
        print("\n→ Smaller R (tighter curvature) raises radial |∇ρ| at contact rim.")
    elif r_lo.grad_ratio_radial > r_hi.grad_ratio_radial:
        print("\n→ Radial gradient decreases with R (approaches flat parallel-plate limit).")

    plot_scan(ch, results, args.d_min, args.alpha_max, OUTPUT / "ch_gpe_sphere_plate.png")
    write_report(ch, results, args.d_min, args.alpha_max, OUTPUT / "ch_gpe_sphere_plate.txt")


if __name__ == "__main__":
    main()
