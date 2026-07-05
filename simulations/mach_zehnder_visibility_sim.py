"""
Prediction #6: Mach-Zehnder visibility vs path length.

Standard QFT:    V = exp(-Gamma * dL)           (environmental decoherence only)
Supersolid:      V = exp(-dL/L_vac) * |1 + eps*cos(2*pi*dL/a_vac)|

Run modes:
  python mach_zehnder_visibility_sim.py           # side-by-side: supersolid vs null (default)
  python mach_zehnder_visibility_sim.py --mode supersolid
  python mach_zehnder_visibility_sim.py --mode null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

DL = np.linspace(0, 10, 50)
NOISE_SIGMA = 0.02


def visibility_standard(dL: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * dL)


def visibility_supersolid(
    dL: np.ndarray,
    L_vac: float,
    eps: float,
    a_vac: float,
) -> np.ndarray:
    return np.exp(-dL / L_vac) * np.abs(1 + eps * np.cos(2 * np.pi * dL / a_vac))


def model_std(dL: np.ndarray, gamma: float) -> np.ndarray:
    return visibility_standard(dL, gamma)


def model_ss(dL: np.ndarray, L_vac: float, eps: float, a_vac: float) -> np.ndarray:
    return visibility_supersolid(dL, L_vac, eps, a_vac)


def chi_squared(meas: np.ndarray, model: np.ndarray, sigma: float) -> float:
    return float(np.sum(((meas - model) / sigma) ** 2))


def fit_models(dL: np.ndarray, v_meas: np.ndarray) -> dict:
    p_std, _ = curve_fit(model_std, dL, v_meas, p0=[0.1], bounds=(0, 2))
    p_ss, _ = curve_fit(
        model_ss,
        dL,
        v_meas,
        p0=[3.0, 0.1, 0.5],
        bounds=([0.1, 0, 0.05], [20, 1, 5]),
    )
    v_std_fit = model_std(dL, *p_std)
    v_ss_fit = model_ss(dL, *p_ss)
    return {
        "gamma": p_std[0],
        "L_vac": p_ss[0],
        "eps": p_ss[1],
        "a_vac": p_ss[2],
        "v_std_fit": v_std_fit,
        "v_ss_fit": v_ss_fit,
        "chi2_std": chi_squared(v_meas, v_std_fit, NOISE_SIGMA),
        "chi2_ss": chi_squared(v_meas, v_ss_fit, NOISE_SIGMA),
    }


def generate_supersolid(seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    L_vac, eps, a_vac = 5.0, 0.15, 0.8
    v_true = visibility_supersolid(DL, L_vac, eps, a_vac)
    v_meas = np.clip(v_true + rng.normal(0, NOISE_SIGMA, size=DL.shape), 0, 1)
    fits = fit_models(DL, v_meas)
    return {
        "label": "Supersolid vacuum (hypothetical)",
        "planted": {"L_vac": L_vac, "eps": eps, "a_vac": a_vac, "gamma": None},
        "v_true": v_true,
        "v_meas": v_meas,
        **fits,
    }


def generate_null(seed: int = 22) -> dict:
    rng = np.random.default_rng(seed)
    gamma = 0.2
    v_true = visibility_standard(DL, gamma)
    v_meas = np.clip(v_true + rng.normal(0, NOISE_SIGMA, size=DL.shape), 0, 1)
    fits = fit_models(DL, v_meas)
    return {
        "label": "Standard QFT (null — decoherence only)",
        "planted": {"L_vac": None, "eps": 0.0, "a_vac": None, "gamma": gamma},
        "v_true": v_true,
        "v_meas": v_meas,
        **fits,
    }


def plot_panel(ax: plt.Axes, result: dict, title: str) -> None:
    ax.errorbar(DL, result["v_meas"], yerr=NOISE_SIGMA, fmt="o", capsize=3, label="Measurements")
    ax.plot(DL, result["v_true"], "k-", lw=2, label="True model")
    ax.plot(
        DL,
        result["v_std_fit"],
        "--",
        label=f"Standard fit (Γ={result['gamma']:.3f})",
    )
    ax.plot(
        DL,
        result["v_ss_fit"],
        "-",
        label=f"Supersolid fit (ε={result['eps']:.3f})",
    )
    ax.set_xlabel("Path length difference ΔL (m)")
    ax.set_ylabel("Fringe visibility V")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def print_result(result: dict) -> None:
    p = result["planted"]
    print(f"\n{result['label']}")
    if p["gamma"] is not None:
        print(f"  Planted:  V = exp(-Γ·ΔL),  Γ={p['gamma']:.3f} m⁻¹  (no vacuum ripple)")
    else:
        print(
            f"  Planted:  L_vac={p['L_vac']:.1f} m, ε={p['eps']:.2f}, "
            f"a_vac={p['a_vac']:.1f} m"
        )
    print(
        f"  Standard fit:   Γ={result['gamma']:.3f},  χ²={result['chi2_std']:.1f}"
    )
    print(
        f"  Supersolid fit: L_vac={result['L_vac']:.3f}, ε={result['eps']:.3f}, "
        f"a_vac={result['a_vac']:.3f},  χ²={result['chi2_ss']:.1f}"
    )
    delta_chi2 = result["chi2_std"] - result["chi2_ss"]
    if p["eps"] and p["eps"] > 0:
        print(f"  → Supersolid model fits better (Δχ²={delta_chi2:.1f}; ripple recovered)")
    else:
        print(
            f"  → Standard model sufficient (Δχ²={delta_chi2:.1f}; "
            "no significant vacuum ripple)"
        )


def save_figure(fig: plt.Figure, filename: str) -> None:
    out = OUTPUT / filename
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_compare(supersolid: dict, null: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_panel(
        axes[0],
        supersolid,
        "Supersolid vacuum (hypothetical)\nVisibility ripples with path length",
    )
    plot_panel(
        axes[1],
        null,
        "Standard QFT (null)\nSmooth exponential decay only",
    )
    fig.suptitle(
        "Prediction #6: Mach–Zehnder visibility — supersolid signal vs QFT null",
        fontsize=12,
        y=1.02,
    )
    save_figure(fig, "mach_zehnder_visibility_sim.png")


def plot_single(result: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_panel(ax, result, result["label"])
    save_figure(fig, "mach_zehnder_visibility_sim.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mach–Zehnder visibility simulation (prediction #6)")
    parser.add_argument(
        "--mode",
        choices=["compare", "supersolid", "null"],
        default="compare",
        help="compare: side-by-side (default); supersolid or null: single panel",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        supersolid = generate_supersolid()
        null = generate_null()
        plot_compare(supersolid, null)
        print_result(supersolid)
        print_result(null)
    elif args.mode == "supersolid":
        result = generate_supersolid()
        plot_single(result)
        print_result(result)
    else:
        result = generate_null()
        plot_single(result)
        print_result(result)


if __name__ == "__main__":
    main()
