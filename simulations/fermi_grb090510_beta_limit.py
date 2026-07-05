#!/usr/bin/env python3
"""
Real Fermi GRB 090510 dispersion analysis — actual β limit in SI.

Downloads GBM Time-Tagged Event (TTE) data from HEASARC for trigger bn090510016
(GRB 090510, z ≈ 0.903), fits quadratic vacuum dispersion:

    t_rel = t - t_ref = (D_L / c³) · β · E_J²

where E_J is photon energy in joules, D_L is luminosity distance, t_rel is
arrival time relative to the trigger (MET).

Reports:
  • β_fit, β_err (SI: s·J⁻² scaled via slope)
  • 95% CL upper limit β_95
  • Equivalent E_QG,2 (GeV) for comparison with literature

NOTE: Tight published limits on GRB 090510 use Fermi LAT GeV photons.
GBM BGO data (≈0.1–40 MeV) here is REAL but gives a looser, systematic-limited
bound (intrinsic spectral lag dominates). See script output for caveats.

Requirements:
  pip install numpy scipy matplotlib astropy astroquery

  python fermi_grb090510_beta_limit.py
  python fermi_grb090510_beta_limit.py --no-download   # use cached FITS
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy import stats

OUTPUT = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data" / "fermi_grb090510"

# GRB 090510 (bn090510016) — spectroscopic redshift
Z_GRB = 0.903
TRIGGER_NAME = "bn090510016"

C = 299792458.0  # m/s
MEV_TO_J = 1.602176634e-13  # 1 MeV in joules

# HEASARC GBM TTE — BGO b1 (highest-energy GBM detector for this burst)
TTE_URL = (
    "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/2009/"
    f"{TRIGGER_NAME}/current/glg_tte_b1_{TRIGGER_NAME}_v00.fit"
)
TTE_FILE = DATA_DIR / f"glg_tte_b1_{TRIGGER_NAME}_v00.fit"

# Analysis cuts (GBM EBOUNDS are in keV)
T_MAX_S = 0.5          # seconds after trigger (reduce spectral-lag contamination)
E_MIN_KEV = 5000.0     # 5 MeV — BGO high-energy tail
E_MAX_KEV = 40000.0    # 40 MeV — physical BGO upper range


def download_tte(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TTE_FILE.exists() and not force:
        print(f"Using cached {TTE_FILE}")
        return TTE_FILE
    print(f"Downloading {TTE_URL}")
    urllib.request.urlretrieve(TTE_URL, TTE_FILE)
    print(f"Saved {TTE_FILE} ({TTE_FILE.stat().st_size / 1e6:.1f} MB)")
    return TTE_FILE


def load_photons(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return times (s rel. trigger), energies (MeV), trigger MET."""
    with fits.open(path) as hdul:
        trig = float(hdul[0].header["TRIGTIME"])
        ebounds = hdul["EBOUNDS"].data
        events = hdul["EVENTS"].data

    e_min = ebounds["E_MIN"].astype(float)  # keV
    e_max = ebounds["E_MAX"].astype(float)  # keV
    good_chan = (e_max > 0) & (e_max <= 50000) & (e_min >= 0)
    e_center_kev = np.zeros(len(ebounds))
    e_center_kev[good_chan] = 0.5 * (e_min[good_chan] + e_max[good_chan])

    times = events["TIME"].astype(float) - trig
    pha = events["PHA"].astype(int)

    valid = (
        (pha >= 0)
        & (pha < len(e_center_kev))
        & good_chan[pha]
        & (times >= 0)
        & (times <= T_MAX_S)
    )
    times = times[valid]
    energies_mev = e_center_kev[pha[valid]] / 1000.0  # keV → MeV

    m = (e_center_kev[pha[valid]] >= E_MIN_KEV) & (e_center_kev[pha[valid]] <= E_MAX_KEV)
    return times[m], energies_mev[m], trig


def luminosity_distance_m(z: float) -> float:
    d_l = Planck18.luminosity_distance(z)
    return float(d_l.to(u.m).value)


def fit_beta(times_s: np.ndarray, energies_mev: np.ndarray, d_l_m: float) -> dict:
    """
    Linear fit: t = a + slope * E_J²
    slope = β * D_L / c³  =>  β = slope * c³ / D_L
    """
    e_j = energies_mev * MEV_TO_J
    x = e_j**2
    y = times_s

    res = stats.linregress(x, y)
    slope = res.slope
    slope_err = res.stderr
    beta = slope * C**3 / d_l_m
    beta_err = slope_err * C**3 / d_l_m

    # 95% CL one-sided upper limit (Gaussian; conservative for demonstration)
    beta_95 = beta + 1.96 * beta_err

    # Equivalent E_QG,2 (quadratic LV scale) from β_95:
    # Common mapping: β ≈ (3/2) * c² * (1+z)² / E_QG_J²  (order-of-magnitude; see paper)
    e_qg_j = np.sqrt(1.5 * C**2 * (1 + Z_GRB) ** 2 / max(beta_95, 1e-30))
    e_qg_gev = e_qg_j / (1e9 * MEV_TO_J)

    chi2 = np.sum((y - (res.intercept + slope * x)) ** 2)
    dof = max(len(y) - 2, 1)
    chi2_red = chi2 / dof

    return {
        "n_photons": len(y),
        "beta": beta,
        "beta_err": beta_err,
        "beta_95": beta_95,
        "intercept_s": res.intercept,
        "slope": slope,
        "slope_err": slope_err,
        "r_value": res.rvalue,
        "p_value": res.pvalue,
        "chi2_red": chi2_red,
        "e_qg_gev": e_qg_gev,
        "d_l_m": d_l_m,
        "d_l_gpc": d_l_m / 3.086e25,
    }


def plot_result(times: np.ndarray, energies: np.ndarray, fit: dict, out: Path) -> None:
    e_j = energies * MEV_TO_J
    x = e_j**2
    t_fit = fit["intercept_s"] + fit["slope"] * x

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(energies, times * 1e3, s=12, alpha=0.7, label=f"GBM BGO photons (n={fit['n_photons']})")
    e_grid = np.linspace(energies.min(), energies.max(), 100)
    x_grid = (e_grid * MEV_TO_J) ** 2
    axes[0].plot(e_grid, (fit["intercept_s"] + fit["slope"] * x_grid) * 1e3, "r-", lw=2, label="Linear fit in E²")
    axes[0].set_xlabel("Photon energy (MeV)")
    axes[0].set_ylabel("Arrival time rel. trigger (ms)")
    axes[0].set_title(f"GRB 090510 — real Fermi GBM data\nwindow: 0–{T_MAX_S}s, E ∈ [{E_MIN_KEV/1e3:.0f}, {E_MAX_KEV/1e3:.0f}] MeV")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x, times * 1e3, s=12, alpha=0.7)
    axes[1].plot(x, t_fit * 1e3, "r-", lw=2)
    axes[1].set_xlabel(r"$E^2$ (J²)")
    axes[1].set_ylabel("Arrival time (ms)")
    axes[1].set_title(r"Fit: $t = a + (\beta D_L/c^3) E^2$")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Quadratic dispersion fit | β = {fit['beta']:.2e} ± {fit['beta_err']:.2e} SI\n"
        f"95% upper limit β < {fit['beta_95']:.2e}  →  E_QG,2 ≳ {fit['e_qg_gev']:.2e} GeV",
        fontsize=10,
        y=1.05,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_report(fit: dict) -> None:
    print("\n" + "=" * 72)
    print("GRB 090510 — REAL FERMI DATA — QUADRATIC DISPERSION LIMIT")
    print("=" * 72)
    print(f"  Trigger:      {TRIGGER_NAME}")
    print(f"  Redshift:     z = {Z_GRB}")
    print(f"  D_L:          {fit['d_l_gpc']:.3f} Gpc  ({fit['d_l_m']:.3e} m)")
    print(f"  Photons used: {fit['n_photons']} (GBM BGO, {E_MIN_KEV/1e3:.0f}–{E_MAX_KEV/1e3:.0f} MeV, first {T_MAX_S}s)")
    print()
    print("  Dispersion model (SI):")
    print("    t = (D_L / c³) · β · E_J²     with E_J in joules")
    print()
    print(f"  β_fit  = {fit['beta']:.4e} ± {fit['beta_err']:.4e}")
    print(f"  β_95   < {fit['beta_95']:.4e}   (95% CL upper limit, Gaussian)")
    print(f"  E_QG,2 ≳ {fit['e_qg_gev']:.3e} GeV   (mapped from β_95; see note below)")
    print()
    print(f"  Fit quality: r = {fit['r_value']:.4f}, p = {fit['p_value']:.3e}, χ²_red = {fit['chi2_red']:.2f}")
    print()
    print("  Literature (Fermi LAT GeV photons, same burst):")
    print("    E_QG,2 ≳ 10¹⁰–10¹¹ GeV  (Amelino-Camelia et al.; Vasileiou et al.)")
    print()
    print("  CH interpretation:")
    print(f"    Strong vacuum dispersion with β ≳ {fit['beta_95']:.2e} is excluded at this distance.")
    print("    CH models with β well below this bound remain consistent.")
    print()
    print("  CAVEATS:")
    print("    • GBM MeV photons — NOT the GeV LAT sample used for tight published limits.")
    print("    • Intrinsic spectral lag mimics dispersion; this is a DEMO real-data pipeline.")
    print("    • For publication-quality limits, use LAT LLE/Extended photons > 1 GeV.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="GRB 090510 real β limit from Fermi GBM data")
    parser.add_argument("--no-download", action="store_true", help="Use cached FITS only")
    parser.add_argument("--force-download", action="store_true", help="Re-download FITS")
    args = parser.parse_args()

    if not args.no_download:
        download_tte(force=args.force_download)
    elif not TTE_FILE.exists():
        raise SystemExit(f"No cached file at {TTE_FILE}; run without --no-download")

    times, energies, _ = load_photons(TTE_FILE)
    if len(times) < 10:
        raise SystemExit(f"Too few photons ({len(times)}); check cuts or download.")

    d_l = luminosity_distance_m(Z_GRB)
    fit = fit_beta(times, energies, d_l)

    OUTPUT.mkdir(exist_ok=True)
    plot_result(times, energies, fit, OUTPUT / "fermi_grb090510_beta_real.png")

    # Save numeric results
    summary = OUTPUT / "fermi_grb090510_beta_real.txt"
    with summary.open("w") as f:
        for k, v in fit.items():
            f.write(f"{k}={v}\n")
    print(f"Saved {summary}")

    print_report(fit)


if __name__ == "__main__":
    main()
