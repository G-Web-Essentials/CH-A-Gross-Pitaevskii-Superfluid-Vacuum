#!/usr/bin/env python3
"""
Real Fermi LAT LLE GRB 090510 dispersion analysis — GeV β limit in SI.

Downloads LAT Low-Energy Events (LLE) for trigger bn090510016 from HEASARC:
  .../fermi/data/lat/triggers/2009/bn090510016/current/gll_lle_bn090510016_v01.fit

Fits quadratic vacuum dispersion (same convention as GBM script):
    t_rel = (D_L / c³) · β · E_J²

Default cuts: prompt window 0–10 s, E ≥ 1 GeV (LAT GeV subsample).

NOTE: The famous 31 GeV photon (Abdo et al. 2009) comes from extended LAT
analysis (Fermi Science Tools on Pass 7 photons), not the pre-cut LLE file.
This LLE file reaches E_max ≈ 4 GeV — tighter than GBM MeV, looser than the
full published LAT dispersion limits (E_QG,2 ~ 10¹⁰–10¹¹ GeV).

Requirements:
  pip install numpy scipy matplotlib astropy

  python fermi_grb090510_lat_lle_beta_limit.py
  python fermi_grb090510_lat_lle_beta_limit.py --no-download
  python fermi_grb090510_lat_lle_beta_limit.py --e-min-gev 0.1   # more photons
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ch_dispersion_core import BETA_FERMI_BOUND, E_QG_FERMI_GEV
from fermi_grb_beta_utils import (
    GEV_TO_J,
    TRIGGER_NAME,
    Z_GRB,
    fit_beta,
    luminosity_distance_m,
)

OUTPUT = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data" / "fermi_grb090510"

LLE_VERSION = "v01"
LLE_BASE = (
    "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/triggers/2009/"
    f"{TRIGGER_NAME}/current"
)
LLE_URL = f"{LLE_BASE}/gll_lle_{TRIGGER_NAME}_{LLE_VERSION}.fit"
LLE_FILE = DATA_DIR / f"gll_lle_{TRIGGER_NAME}_{LLE_VERSION}.fit"

# Analysis defaults (literature prompt + GeV band)
T_MIN_S = 0.0
T_MAX_S = 10.0
E_MIN_GEV = 1.0
E_MAX_GEV = 100.0


def download_lle(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LLE_FILE.exists() and not force:
        print(f"Using cached {LLE_FILE}")
        return LLE_FILE
    print(f"Downloading {LLE_URL}")
    try:
        urllib.request.urlretrieve(LLE_URL, LLE_FILE)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Download failed ({exc.code}): {LLE_URL}") from exc
    print(f"Saved {LLE_FILE} ({LLE_FILE.stat().st_size / 1e6:.1f} MB)")
    return LLE_FILE


def load_lle_photons(path: Path, t_min: float, t_max: float, e_min_gev: float, e_max_gev: float) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Return times (s rel. trigger), energies (GeV), trigger MET, metadata."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        trig = float(hdr["TRIGTIME"])
        ev = hdul["EVENTS"].data
        meta = {
            "object": hdr.get("OBJECT", TRIGGER_NAME),
            "llecut": str(hdr.get("LLECUT", ""))[:120],
            "tstart": float(hdr.get("TSTART", 0)),
            "tstop": float(hdr.get("TSTOP", 0)),
            "n_events_file": len(ev),
        }

    times = ev["TIME"].astype(float) - trig
    energies_gev = ev["ENERGY"].astype(float) / 1000.0  # MeV in FITS → GeV

    valid = (
        (times >= t_min)
        & (times <= t_max)
        & (energies_gev >= e_min_gev)
        & (energies_gev <= e_max_gev)
    )
    return times[valid], energies_gev[valid], trig, meta


def plot_result(
    times: np.ndarray,
    energies_gev: np.ndarray,
    fit: dict,
    out: Path,
    t_max: float,
    e_min: float,
) -> None:
    e_j = energies_gev * GEV_TO_J
    x = e_j**2
    t_fit = fit["intercept_s"] + fit["slope"] * x

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(energies_gev, times * 1e3, s=28, alpha=0.85, c="#1f77b4", edgecolors="k", linewidths=0.3,
                    label=f"LAT LLE photons (n={fit['n_photons']})")
    if len(energies_gev) >= 2:
        e_grid = np.linspace(energies_gev.min(), energies_gev.max(), 100)
        x_grid = (e_grid * GEV_TO_J) ** 2
        axes[0].plot(e_grid, (fit["intercept_s"] + fit["slope"] * x_grid) * 1e3, "r-", lw=2, label="Fit in E²")
    axes[0].set_xlabel("Photon energy (GeV)")
    axes[0].set_ylabel("Arrival time rel. trigger (ms)")
    axes[0].set_title(
        f"GRB 090510 — real Fermi LAT LLE\n"
        f"window: {T_MIN_S}–{t_max}s, E ≥ {e_min} GeV"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x, times * 1e3, s=28, alpha=0.85, c="#1f77b4", edgecolors="k", linewidths=0.3)
    if len(x) >= 2:
        axes[1].plot(x, t_fit * 1e3, "r-", lw=2)
    axes[1].set_xlabel(r"$E^2$ (J²)")
    axes[1].set_ylabel("Arrival time (ms)")
    axes[1].set_title(r"Fit: $t = a + (\beta D_L/c^3) E^2$")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"LAT LLE dispersion | β = {fit['beta']:.2e} ± {fit['beta_err']:.2e}\n"
        f"β₉₅ = {fit['beta_95']:.2e}  →  E_QG,2 ≳ {fit['e_qg_gev']:.2e} GeV  "
        f"(literature LAT: ≳ {E_QG_FERMI_GEV:.0e} GeV)",
        fontsize=10,
        y=1.06,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_report(fit: dict, meta: dict, t_max: float, e_min: float, e_max: float) -> None:
    print("\n" + "=" * 72)
    print("GRB 090510 — REAL FERMI LAT LLE — QUADRATIC DISPERSION LIMIT")
    print("=" * 72)
    print(f"  Trigger:      {TRIGGER_NAME}  ({meta.get('object', '')})")
    print(f"  Redshift:     z = {Z_GRB}")
    print(f"  D_L:          {fit['d_l_gpc']:.3f} Gpc")
    print(f"  File events:  {meta.get('n_events_file', '?')} total in LLE FITS")
    print(f"  Photons used: {fit['n_photons']} (LLE, {e_min}–{e_max} GeV, {T_MIN_S}–{t_max}s)")
    print()
    print("  Dispersion model (SI):")
    print("    t = (D_L / c³) · β · E_J²")
    print()
    print(f"  β_fit  = {fit['beta']:.4e} ± {fit['beta_err']:.4e}")
    print(f"  β_95   < {fit['beta_95']:.4e}   (95% CL, Gaussian)")
    print(f"  E_QG,2 ≳ {fit['e_qg_gev']:.3e} GeV")
    print()
    print(f"  Fit quality: r = {fit['r_value']:.4f}, p = {fit['p_value']:.3e}, χ²_red = {fit['chi2_red']:.2f}")
    print()
    print("  Literature (extended LAT, incl. 31 GeV photon):")
    print(f"    E_QG,2 ≳ 10¹⁰–10¹¹ GeV  →  β ≲ {BETA_FERMI_BOUND:.2e}")
    print()
    print("  CH interpretation:")
    print("    Null / weak fit → consistent with gradient-gated CH (β_eff ≈ 0 in void).")
    print("    Naive always-on CH (β = β_max) remains excluded by literature limits.")
    print()
    print("  CAVEATS:")
    print("    • LLE is pre-selected Pass-7 events (E_max ~ few GeV in this file).")
    print("    • Published tight limits use extended LAT photons up to 31 GeV.")
    print("    • Intrinsic spectral lag still mimics dispersion at these energies.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRB 090510 β limit from Fermi LAT LLE")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--t-max", type=float, default=T_MAX_S, help="Max time after trigger [s]")
    parser.add_argument("--e-min-gev", type=float, default=E_MIN_GEV, help="Min photon energy [GeV]")
    parser.add_argument("--e-max-gev", type=float, default=E_MAX_GEV, help="Max photon energy [GeV]")
    args = parser.parse_args()

    if not args.no_download:
        download_lle(force=args.force_download)
    elif not LLE_FILE.exists():
        raise SystemExit(f"No cached file at {LLE_FILE}; run without --no-download")

    times, energies, _, meta = load_lle_photons(
        LLE_FILE, T_MIN_S, args.t_max, args.e_min_gev, args.e_max_gev
    )
    if len(times) < 5:
        raise SystemExit(
            f"Too few photons ({len(times)}) with E ≥ {args.e_min_gev} GeV; "
            f"try --e-min-gev 0.1 or widen --t-max"
        )

    d_l = luminosity_distance_m()
    fit = fit_beta(times, energies * GEV_TO_J, d_l)

    OUTPUT.mkdir(exist_ok=True)
    plot_result(times, energies, fit, OUTPUT / "fermi_grb090510_lat_lle_beta_real.png", args.t_max, args.e_min_gev)

    summary = OUTPUT / "fermi_grb090510_lat_lle_beta_real.txt"
    with summary.open("w") as f:
        f.write(f"source=LAT_LLE\n")
        f.write(f"lle_file={LLE_FILE.name}\n")
        f.write(f"t_min={T_MIN_S}\n")
        f.write(f"t_max={args.t_max}\n")
        f.write(f"e_min_gev={args.e_min_gev}\n")
        f.write(f"e_max_gev={args.e_max_gev}\n")
        for k, v in fit.items():
            f.write(f"{k}={v}\n")
    print(f"Saved {summary}")

    print_report(fit, meta, args.t_max, args.e_min_gev, args.e_max_gev)


if __name__ == "__main__":
    main()
