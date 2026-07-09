"""
Paper 2 — export GPE Casimir-gap profiles for EM coupling (Phase 0).

Writes scan tables and per-gap density profiles used by ch_casimir_em_coupling.py.

  python ch_paper2_gpe_export.py
  python ch_paper2_gpe_export.py --xi 50e-9 --d-min 40e-9 --d-max 600e-9
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_core import (
    GPE1DResult,
    scan_gap_separations,
    thomas_fermi_box_profile,
)

DATA = Path(__file__).parent / "data" / "paper2"
OUTPUT = Path(__file__).parent / "output" / "paper2"


def export_scan_csv(ch: CHParams, results: list[GPE1DResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "d_m",
                "d_nm",
                "d_hat",
                "k_per_m",
                "k_per_nm",
                "xi_m",
                "rho_in",
                "grad_ratio_wall",
                "grad_ratio_mid",
                "chi_wall",
                "chi_bulk",
            ]
        )
        for r in results:
            w.writerow(
                [
                    f"{r.d_m:.6e}",
                    f"{r.d_m * 1e9:.6f}",
                    f"{r.d_hat:.6e}",
                    f"{1.0 / r.d_m:.6e}",
                    f"{1.0 / (r.d_m * 1e9):.6e}",
                    f"{r.xi_m:.6e}",
                    f"{r.rho_in:.6e}",
                    f"{r.grad_ratio_wall:.6e}",
                    f"{r.grad_ratio_mid:.6e}",
                    f"{r.chi_wall:.6e}",
                    f"{r.chi_mid:.6e}",
                ]
            )
    print(f"Wrote {path} ({len(results)} rows)")


def export_profile_csv(
    ch: CHParams,
    d_m: float,
    path: Path,
    n_points: int = 4000,
) -> None:
    """ρ(ẑ) on a ξ-resolved grid for one gap (Thomas–Fermi analytic)."""
    d_hat = d_m / ch.xi
    path.parent.mkdir(parents=True, exist_ok=True)
    z_hat = np.linspace(0.0, d_hat, n_points)
    rho = thomas_fermi_box_profile(z_hat, d_hat)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_m", "d_nm", "d_hat", "xi_m", "z_hat", "z_m", "rho_norm", "depletion"])
        for zh, rh in zip(z_hat, rho):
            w.writerow(
                [
                    f"{d_m:.6e}",
                    f"{d_m * 1e9:.6f}",
                    f"{d_hat:.6e}",
                    f"{ch.xi:.6e}",
                    f"{zh:.8e}",
                    f"{zh * ch.xi:.8e}",
                    f"{rh:.8e}",
                    f"{1.0 - rh:.8e}",
                ]
            )
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Paper 2 GPE Casimir-gap tables")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=40e-9, help="Min gap [m]")
    parser.add_argument("--d-max", type=float, default=600e-9, help="Max gap [m]")
    parser.add_argument("--n-d", type=int, default=40, help="Gap scan points")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_array = np.logspace(np.log10(args.d_min), np.log10(args.d_max), args.n_d)
    results = scan_gap_separations(ch, d_array)

    tag = f"xi{int(args.xi * 1e9)}nm"
    export_scan_csv(ch, results, DATA / f"gpe_scan_{tag}.csv")

    # Representative profiles: small, turn-on, large gap
    for label, d_m in [
        ("d40nm", 40e-9),
        ("d100nm", 100e-9),
        ("d600nm", 600e-9),
    ]:
        export_profile_csv(ch, d_m, DATA / f"rho_profile_{tag}_{label}.csv")

    # Copy summary to output for plots
    export_scan_csv(ch, results, OUTPUT / f"gpe_scan_{tag}.csv")


if __name__ == "__main__":
    main()
