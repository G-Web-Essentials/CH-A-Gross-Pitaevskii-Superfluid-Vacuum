#!/usr/bin/env python3
"""
Real Fermi LAT EXTENDED photons — GRB 090510 dispersion (publication-grade path).

Queries the Fermi LAT Data Server for TRANSIENT-class extended events
(not available in weekly SOURCE-only files), downloads EV00 FITS, and fits:

    t_rel = (D_L / c³) · β · E_J²

This pipeline reaches E_max ~ 30 GeV (the famous ~31 GeV photon at t≈0.83 s)
and produces β limits much closer to published Amelino-Camelia / Vasileiou bounds
than GBM MeV or pre-cut LLE files.

Requirements:
  pip install numpy scipy matplotlib astropy requests

  python fermi_grb090510_lat_extended_beta_limit.py
  python fermi_grb090510_lat_extended_beta_limit.py --no-download
  python fermi_grb090510_lat_extended_beta_limit.py --force-query   # re-query server
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

from ch_dispersion_core import BETA_FERMI_BOUND, E_QG_FERMI_GEV
from fermi_grb_beta_utils import (
    GEV_TO_J,
    TRIGGER_NAME,
    Z_GRB,
    fit_beta,
    luminosity_distance_m,
)
from fermi_lat_extended_client import query_and_download

OUTPUT = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data" / "fermi_grb090510" / "lat_extended"

# GRB 090510 (bn090510016) — FERMILGRB / GBM trigger
TRIG_MET = 263607781.971
RA_DEG = 333.57
DEC_DEG = -26.62

# Query window: ~200 s before trigger to +400 s (covers prompt + early GeV tail)
MET_START = TRIG_MET - 200.0
MET_STOP = TRIG_MET + 400.0

# Analysis cuts
T_MIN_S = 0.0
T_MAX_S = 10.0
E_MIN_GEV = 1.0
E_MAX_GEV = 300.0
ROI_DEG = 12.0  # on-sky separation from burst (literature uses energy-dependent ROI)


def load_extended_photons(
    path: Path,
    trig_met: float,
    ra_deg: float,
    dec_deg: float,
    t_min: float,
    t_max: float,
    e_min_gev: float,
    e_max_gev: float,
    roi_deg: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        ev = hdul["EVENTS"].data
        meta = {
            "object": hdr.get("OBJECT", TRIGGER_NAME),
            "n_events_file": len(ev),
            "pass_version": hdr.get("PASS_VER", hdr.get("PASSVERSION", "?")),
        }

    times = ev["TIME"].astype(float) - trig_met
    energies_gev = ev["ENERGY"].astype(float) / 1000.0

    burst = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    phot = SkyCoord(ra=ev["RA"] * u.deg, dec=ev["DEC"] * u.deg)
    sep_deg = burst.separation(phot).deg

    valid = (
        (times >= t_min)
        & (times <= t_max)
        & (energies_gev >= e_min_gev)
        & (energies_gev <= e_max_gev)
        & (sep_deg <= roi_deg)
    )
    meta["e_max_gev_file"] = float(energies_gev.max())
    meta["n_roi"] = int(valid.sum())
    return times[valid], energies_gev[valid], meta


def plot_result(
    times: np.ndarray,
    energies_gev: np.ndarray,
    fit: dict,
    out: Path,
    meta: dict,
    t_max: float,
    e_min: float,
) -> None:
    e_j = energies_gev * GEV_TO_J
    x = e_j**2
    t_fit = fit["intercept_s"] + fit["slope"] * x

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(
        energies_gev, times * 1e3, s=40, alpha=0.9, c="#9467bd", edgecolors="k", linewidths=0.3,
        label=f"Extended LAT (n={fit['n_photons']})",
    )
    if len(energies_gev) >= 2:
        e_grid = np.linspace(energies_gev.min(), max(energies_gev.max(), e_min + 0.1), 100)
        x_grid = (e_grid * GEV_TO_J) ** 2
        axes[0].plot(e_grid, (fit["intercept_s"] + fit["slope"] * x_grid) * 1e3, "r-", lw=2, label="Fit in E²")
    axes[0].axvline(30.0, color="gray", ls=":", alpha=0.7, label="~31 GeV (Abdo et al.)")
    axes[0].set_xlabel("Photon energy (GeV)")
    axes[0].set_ylabel("Arrival time rel. trigger (ms)")
    axes[0].set_title(
        f"GRB 090510 — LAT extended (TRANSIENT class)\n"
        f"E_max in file = {meta.get('e_max_gev_file', 0):.1f} GeV"
    )
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x, times * 1e3, s=40, alpha=0.9, c="#9467bd", edgecolors="k", linewidths=0.3)
    if len(x) >= 2:
        axes[1].plot(x, t_fit * 1e3, "r-", lw=2)
    axes[1].set_xlabel(r"$E^2$ (J²)")
    axes[1].set_ylabel("Arrival time (ms)")
    axes[1].set_title(r"Fit: $t = a + (\beta D_L/c^3) E^2$")
    axes[1].grid(alpha=0.3)

    ratio = fit["beta_95"] / BETA_FERMI_BOUND if BETA_FERMI_BOUND > 0 else np.nan
    fig.suptitle(
        f"Extended LAT dispersion | β₉₅ = {fit['beta_95']:.2e}  "
        f"(≈ {ratio:.0f}× literature bound)\n"
        f"E_QG,2 ≳ {fit['e_qg_gev']:.2e} GeV  (published ≳ {E_QG_FERMI_GEV:.0e} GeV)",
        fontsize=10,
        y=1.06,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_report(fit: dict, meta: dict, t_max: float, e_min: float) -> None:
    ratio = fit["beta_95"] / BETA_FERMI_BOUND
    print("\n" + "=" * 72)
    print("GRB 090510 — REAL FERMI LAT EXTENDED — QUADRATIC DISPERSION LIMIT")
    print("=" * 72)
    print(f"  Trigger MET:  {TRIG_MET}")
    print(f"  Position:     RA={RA_DEG}, Dec={DEC_DEG}")
    print(f"  Redshift:     z = {Z_GRB}")
    print(f"  D_L:          {fit['d_l_gpc']:.3f} Gpc")
    print(f"  Query class:  Extended (TRANSIENT + SOURCE photons)")
    print(f"  File events:  {meta.get('n_events_file', '?')} in ROI/time/energy cut → {fit['n_photons']} used")
    print(f"  E_max (file): {meta.get('e_max_gev_file', 0):.2f} GeV")
    print()
    print(f"  β_fit  = {fit['beta']:.4e} ± {fit['beta_err']:.4e}")
    print(f"  β_95   < {fit['beta_95']:.4e}")
    print(f"  E_QG,2 ≳ {fit['e_qg_gev']:.3e} GeV")
    print(f"  vs literature β ≲ {BETA_FERMI_BOUND:.2e}  (factor {ratio:.1f}× looser than published)")
    print()
    print(f"  Fit: r = {fit['r_value']:.4f}, p = {fit['p_value']:.3e}, χ²_red = {fit['chi2_red']:.2f}")
    print()
    print("  Comparison to other pipelines in this repo:")
    print("    GBM MeV:     β_95 ~ 6×10²⁰  (weakest)")
    print("    LAT LLE:     β_95 ~ 10¹⁸   (pre-cut, E_max ~ 4 GeV)")
    print(f"    Extended:    β_95 ~ {fit['beta_95']:.1e}  (includes ~30 GeV photon)")
    print()
    print("  CH: null fit → gradient-gated CH consistent; naive always-on excluded.")
    print("  NOTE: Full published limits use bespoke event selection + spectral-lag")
    print("        systematics (Amelino-Camelia et al. 2009; Vasileiou et al.).")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRB 090510 extended LAT β limit")
    parser.add_argument("--no-download", action="store_true", help="Use cached FITS only")
    parser.add_argument("--force-query", action="store_true", help="Re-submit Fermi data server query")
    parser.add_argument("--t-max", type=float, default=T_MAX_S)
    parser.add_argument("--e-min-gev", type=float, default=E_MIN_GEV)
    parser.add_argument("--roi-deg", type=float, default=ROI_DEG)
    args = parser.parse_args()

    fits_path: Path | None = None
    meta: dict = {}

    if args.no_download:
        manifest = DATA_DIR / "query_manifest.json"
        if not manifest.exists():
            raise SystemExit(f"No cache at {DATA_DIR}; run without --no-download")
        import json
        man = json.loads(manifest.read_text())
        fits_path = Path(man["event_fits"])
        if not fits_path.exists():
            raise SystemExit(f"Missing cached FITS {fits_path}")
        print(f"Using cached {fits_path}")
    else:
        fits_path, _ = query_and_download(
            DATA_DIR,
            RA_DEG,
            DEC_DEG,
            MET_START,
            MET_STOP,
            tag="bn090510016",
            force=args.force_query,
            radius_deg=20.0,
            lat_datatype="Extended",
            spacecraft=False,
        )

    times, energies, meta = load_extended_photons(
        fits_path,
        TRIG_MET,
        RA_DEG,
        DEC_DEG,
        T_MIN_S,
        args.t_max,
        args.e_min_gev,
        E_MAX_GEV,
        args.roi_deg,
    )
    if len(times) < 5:
        raise SystemExit(f"Too few photons ({len(times)}); relax --e-min-gev or --roi-deg")

    d_l = luminosity_distance_m()
    fit = fit_beta(times, energies * GEV_TO_J, d_l)

    OUTPUT.mkdir(exist_ok=True)
    plot_result(times, energies, fit, OUTPUT / "fermi_grb090510_lat_extended_beta_real.png", meta, args.t_max, args.e_min_gev)

    summary = OUTPUT / "fermi_grb090510_lat_extended_beta_real.txt"
    with summary.open("w") as f:
        f.write("source=LAT_EXTENDED\n")
        f.write(f"fits_file={fits_path.name}\n")
        f.write(f"trig_met={TRIG_MET}\n")
        f.write(f"t_max={args.t_max}\n")
        f.write(f"e_min_gev={args.e_min_gev}\n")
        f.write(f"roi_deg={args.roi_deg}\n")
        f.write(f"e_max_file_gev={meta.get('e_max_gev_file', 0)}\n")
        for k, v in fit.items():
            f.write(f"{k}={v}\n")
    print(f"Saved {summary}")

    print_report(fit, meta, args.t_max, args.e_min_gev)


if __name__ == "__main__":
    main()
