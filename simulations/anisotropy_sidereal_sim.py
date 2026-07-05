"""
Prediction #1: Directional anisotropy — sidereal modulation in fractional speed shift.

delta_c/c0 = A * cos(omega_s * t + phi0) + noise

Run modes:
  python anisotropy_sidereal_sim.py           # side-by-side: supersolid vs null (default)
  python anisotropy_sidereal_sim.py --mode supersolid
  python anisotropy_sidereal_sim.py --mode null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

OMEGA_S = 2 * np.pi / (23 * 3600 + 56 * 60)


def sidereal_signal(t_sec: np.ndarray, A: float, phi0: float) -> np.ndarray:
    return A * np.cos(OMEGA_S * t_sec + phi0)


def generate_dataset(
    A_true: float,
    phi0_true: float,
    n_days: int = 14,
    samples_per_day: int = 24,
    noise_std: float = 5e-19,
    seed: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_sec = np.linspace(0, n_days * 86400, n_days * samples_per_day)
    signal = sidereal_signal(t_sec, A_true, phi0_true)
    noise = rng.normal(0, noise_std, size=t_sec.shape)
    meas = signal + noise
    return t_sec, meas, signal


def fit_sidereal(t_sec: np.ndarray, meas: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    def fit_fn(t, A, phi0):
        return sidereal_signal(t, A, phi0)

    popt, pcov = curve_fit(fit_fn, t_sec, meas, p0=[1e-18, 0.0])
    A_fit, phi_fit = popt
    A_err = np.sqrt(pcov[0, 0])
    fit = fit_fn(t_sec, *popt)
    return A_fit, A_err, phi_fit, fit


def plot_panel(
    ax: plt.Axes,
    t_sec: np.ndarray,
    meas: np.ndarray,
    fit: np.ndarray,
    title: str,
    n_days_show: int = 3,
    samples_per_day: int = 24,
) -> None:
    hours = (t_sec % 86400) / 3600
    n_show = n_days_show * samples_per_day
    ax.plot(hours[:n_show], meas[:n_show], ".", alpha=0.6, label="Measurements")
    ax.plot(hours[:n_show], fit[:n_show], "r-", lw=2, label="Sidereal fit")
    ax.set_xlabel("Hour of day (folded)")
    ax.set_ylabel(r"$\delta c / c_0$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def run_supersolid(seed: int = 5) -> dict:
    A_true, phi0_true = 2e-18, 1.2
    t_sec, meas, signal = generate_dataset(A_true, phi0_true, seed=seed)
    A_fit, A_err, phi_fit, fit = fit_sidereal(t_sec, meas)
    return {
        "label": "Supersolid vacuum (hypothetical signal)",
        "A_true": A_true,
        "phi0_true": phi0_true,
        "t_sec": t_sec,
        "meas": meas,
        "fit": fit,
        "A_fit": A_fit,
        "A_err": A_err,
        "phi_fit": phi_fit,
    }


def run_null(seed: int = 7) -> dict:
    t_sec, meas, _ = generate_dataset(A_true=0.0, phi0_true=0.0, seed=seed)
    A_fit, A_err, phi_fit, fit = fit_sidereal(t_sec, meas)
    return {
        "label": "Standard QFT (null — noise only)",
        "A_true": 0.0,
        "phi0_true": None,
        "t_sec": t_sec,
        "meas": meas,
        "fit": fit,
        "A_fit": A_fit,
        "A_err": A_err,
        "phi_fit": phi_fit,
    }


def print_result(result: dict) -> None:
    print(f"\n{result['label']}")
    if result["A_true"] > 0:
        print(f"  Planted:  A={result['A_true']:.2e}, phi={result['phi0_true']:.3f} rad")
    else:
        print("  Planted:  A=0 (no anisotropy)")
    print(
        f"  Fit:      A={result['A_fit']:.2e} ± {result['A_err']:.2e}, "
        f"phi={result['phi_fit']:.3f} rad"
    )
    sigma = abs(result["A_fit"]) / result["A_err"] if result["A_err"] > 0 else 0.0
    if result["A_true"] > 0:
        print(f"  → Signal recovered ({sigma:.1f}σ above zero)")
    else:
        print(f"  → No significant signal ({sigma:.1f}σ; consistent with flat QFT vacuum)")


def save_figure(fig: plt.Figure, filename: str) -> Path:
    out = OUTPUT / filename
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_compare(supersolid: dict, null: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    plot_panel(
        axes[0],
        supersolid["t_sec"],
        supersolid["meas"],
        supersolid["fit"],
        "Supersolid vacuum (hypothetical)\nSidereal modulation in δc/c₀",
    )
    plot_panel(
        axes[1],
        null["t_sec"],
        null["meas"],
        null["fit"],
        "Standard QFT (null)\nNo preferred direction — noise only",
    )

    fig.suptitle(
        "Prediction #1: Anisotropy test — supersolid signal vs QFT null",
        fontsize=12,
        y=1.02,
    )
    save_figure(fig, "anisotropy_sidereal_sim.png")


def plot_single(result: dict, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_panel(ax, result["t_sec"], result["meas"], result["fit"], result["label"])
    save_figure(fig, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sidereal anisotropy simulation (prediction #1)")
    parser.add_argument(
        "--mode",
        choices=["compare", "supersolid", "null"],
        default="compare",
        help="compare: side-by-side (default); supersolid or null: single panel",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        supersolid = run_supersolid()
        null = run_null()
        plot_compare(supersolid, null)
        print_result(supersolid)
        print_result(null)
    elif args.mode == "supersolid":
        result = run_supersolid()
        plot_single(result, "anisotropy_sidereal_sim.png")
        print_result(result)
    else:
        result = run_null()
        plot_single(result, "anisotropy_sidereal_sim.png")
        print_result(result)


if __name__ == "__main__":
    main()
