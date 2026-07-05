"""
Prediction #3: Casimir force with oscillatory supersolid vacuum correction.

F(d) = F_Cas(d) * [1 + alpha * cos(2*pi*d / a_vac + phi)]

Fits are done on the normalized ratio F/F_Cas to remove the steep 1/d^4 background.

Run modes:
  python casimir_ripple_sim.py           # side-by-side: supersolid vs null (default)
  python casimir_ripple_sim.py --mode supersolid
  python casimir_ripple_sim.py --mode null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

HBAR = 1.054571817e-34
C = 299792458.0
A_PLATE = 1e-6  # m^2

# Plate separation scan: 50–600 nm (covers ~3.7 periods at a_vac=150 nm)
D_M = np.linspace(50e-9, 600e-9, 120)

# Supersolid demo parameters (illustrative)
A_VAC_TRUE = 150e-9  # 150 nm lattice scale
ALPHA_TRUE = 0.12     # 12% ripple — visible but still subtle
PHI_TRUE = 0.3
NOISE_SIGMA_RATIO = 0.004  # 0.4% noise on F/F_Cas


def casimir_force(d: np.ndarray) -> np.ndarray:
    return (np.pi**2 * HBAR * C / 240) * A_PLATE / d**4


def ripple_factor(d: np.ndarray, alpha: float, a_vac: float, phi: float = 0.0) -> np.ndarray:
    return 1.0 + alpha * np.cos(2 * np.pi * d / a_vac + phi)


def casimir_with_ripple(d: np.ndarray, alpha: float, a_vac: float, phi: float = 0.0) -> np.ndarray:
    return casimir_force(d) * ripple_factor(d, alpha, a_vac, phi)


def model_ratio_std(_d, level):
    return np.full_like(_d, level, dtype=float)


def model_ratio_ripple(d, level, alpha, a_vac, phi):
    return level * ripple_factor(d, alpha, a_vac, phi)


def chi2(meas: np.ndarray, model: np.ndarray, sigma: float) -> float:
    return float(np.sum(((meas - model) / sigma) ** 2))


def fit_models(d: np.ndarray, ratio_meas: np.ndarray, sigma: float) -> dict:
    p_std, _ = curve_fit(
        model_ratio_std,
        d,
        ratio_meas,
        p0=[1.0],
        sigma=np.full_like(ratio_meas, sigma),
        absolute_sigma=True,
    )
    ratio_std = model_ratio_std(d, *p_std)

    p_rip, pcov = curve_fit(
        model_ratio_ripple,
        d,
        ratio_meas,
        p0=[1.0, 0.08, A_VAC_TRUE, 0.0],
        sigma=np.full_like(ratio_meas, sigma),
        absolute_sigma=True,
        bounds=([0.9, -0.5, 80e-9, -np.pi], [1.1, 0.5, 250e-9, np.pi]),
    )
    ratio_rip = model_ratio_ripple(d, *p_rip)
    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.zeros(4)

    return {
        "level_std": p_std[0],
        "level_rip": p_rip[0],
        "alpha_fit": p_rip[1],
        "a_vac_fit": p_rip[2],
        "phi_fit": p_rip[3],
        "alpha_err": perr[1] if len(perr) > 1 else float("nan"),
        "a_vac_err": perr[2] if len(perr) > 2 else float("nan"),
        "ratio_std": ratio_std,
        "ratio_rip": ratio_rip,
        "chi2_std": chi2(ratio_meas, ratio_std, sigma),
        "chi2_rip": chi2(ratio_meas, ratio_rip, sigma),
    }


def generate(alpha: float, a_vac: float, phi: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    f_cas = casimir_force(D_M)
    ratio_true = ripple_factor(D_M, alpha, a_vac, phi)
    f_true = f_cas * ratio_true
    ratio_meas = ratio_true + rng.normal(0, NOISE_SIGMA_RATIO, size=D_M.shape)
    f_meas = f_cas * ratio_meas

    fits = fit_models(D_M, ratio_meas, NOISE_SIGMA_RATIO)
    f_std = f_cas * fits["ratio_std"]
    f_rip = f_cas * fits["ratio_rip"]

    return {
        "alpha_true": alpha,
        "a_vac_true": a_vac,
        "phi_true": phi,
        "f_true": f_true,
        "f_meas": f_meas,
        "f_cas": f_cas,
        "ratio_true": ratio_true,
        "ratio_meas": ratio_meas,
        "f_std": f_std,
        "f_rip": f_rip,
        **fits,
    }


def verdict(result: dict, label: str) -> str:
    lines = [f"\n{label}"]
    if result["alpha_true"] > 0:
        lines.append(
            f"  Planted:  α={result['alpha_true']:.3f}, "
            f"a_vac={result['a_vac_true']*1e9:.1f} nm, φ={result['phi_true']:.3f} rad"
        )
    else:
        lines.append("  Planted:  α=0 (standard Casimir only, no vacuum ripple)")
    lines.append(
        f"  Ripple fit: α={result['alpha_fit']:.4f} ± {result['alpha_err']:.4f}, "
        f"a_vac={result['a_vac_fit']*1e9:.1f} ± {result['a_vac_err']*1e9:.1f} nm, "
        f"φ={result['phi_fit']:.3f} rad"
    )
    lines.append(
        f"  χ² standard: {result['chi2_std']:.1f}  |  χ² ripple: {result['chi2_rip']:.1f}"
    )
    delta = result["chi2_std"] - result["chi2_rip"]
    if result["alpha_true"] > 0:
        alpha_ok = abs(result["alpha_fit"] - result["alpha_true"]) < max(3 * result["alpha_err"], 0.03)
        avac_ok = abs(result["a_vac_fit"] - result["a_vac_true"]) < max(3 * result["a_vac_err"], 20e-9)
        if delta > 10 and alpha_ok and avac_ok:
            lines.append(f"  → Ripple recovered (Δχ²={delta:.1f})")
        elif delta > 10:
            lines.append(f"  → Ripple detected but parameters imprecise (Δχ²={delta:.1f})")
        else:
            lines.append(f"  → Weak recovery (Δχ²={delta:.1f})")
        lines.append(
            "  → Real Casimir labs see smooth QED; large α at nm scales would be notable if true"
        )
    else:
        if delta < 10 and abs(result["alpha_fit"]) < 3 * max(result["alpha_err"], 0.01):
            lines.append(f"  → Consistent with null (no ripple; Δχ²={delta:.1f})")
        else:
            lines.append(f"  → Spurious ripple in noise? (Δχ²={delta:.1f})")
    return "\n".join(lines)


def plot_force_panel(ax: plt.Axes, result: dict, title: str) -> None:
    sigma_f = NOISE_SIGMA_RATIO * result["f_cas"]
    ax.errorbar(
        D_M * 1e9,
        result["f_meas"] * 1e12,
        yerr=sigma_f * 1e12,
        fmt=".",
        alpha=0.45,
        label="Measurements",
        zorder=2,
    )
    ax.plot(D_M * 1e9, result["f_true"] * 1e12, "k-", lw=2, label="True", zorder=3)
    ax.plot(D_M * 1e9, result["f_std"] * 1e12, "--", label="Fit: standard Casimir", zorder=4)
    ax.plot(D_M * 1e9, result["f_rip"] * 1e12, "-", label="Fit: Casimir + ripple", zorder=5)
    ax.set_xlabel("Plate separation d (nm)")
    ax.set_ylabel("Force (pN)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_ratio_panel(ax: plt.Axes, result: dict, title: str) -> None:
    ax.errorbar(
        D_M * 1e9,
        result["ratio_meas"],
        yerr=NOISE_SIGMA_RATIO,
        fmt="o",
        ms=3,
        capsize=2,
        label="F / F_Cas data",
        zorder=2,
    )
    ax.plot(D_M * 1e9, result["ratio_true"], "k-", lw=2, label="True ratio", zorder=3)
    ax.plot(D_M * 1e9, result["ratio_std"], "--", label="Standard fit", zorder=4)
    ax.plot(D_M * 1e9, result["ratio_rip"], "-", label="Ripple fit", zorder=5)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Plate separation d (nm)")
    ax.set_ylabel(r"$F / F_{\mathrm{Cas}}$")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def save_figure(fig: plt.Figure, filename: str) -> None:
    out = OUTPUT / filename
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def run_compare() -> None:
    supersolid = generate(ALPHA_TRUE, A_VAC_TRUE, PHI_TRUE, seed=123)
    null = generate(0.0, A_VAC_TRUE, 0.0, seed=456)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    plot_force_panel(
        axes[0, 0],
        supersolid,
        f"Supersolid vacuum (hypothetical)\nα={ALPHA_TRUE}, a_vac={A_VAC_TRUE*1e9:.0f} nm",
    )
    plot_force_panel(axes[0, 1], null, "Standard QFT (null)\nNo vacuum ripple (α=0)")
    plot_ratio_panel(axes[1, 0], supersolid, "Normalized ratio — ripple visible here")
    plot_ratio_panel(axes[1, 1], null, "Normalized ratio — flat at 1.0")
    fig.suptitle(
        "Prediction #3: Casimir ripple — supersolid signal vs QFT null",
        fontsize=12,
        y=1.01,
    )
    save_figure(fig, "casimir_ripple_sim.png")

    print(verdict(supersolid, "Supersolid vacuum (hypothetical)"))
    print(verdict(null, "Standard QFT (null)"))


def run_single(mode: str) -> None:
    if mode == "supersolid":
        result = generate(ALPHA_TRUE, A_VAC_TRUE, PHI_TRUE, seed=123)
        title_f = "Supersolid vacuum (hypothetical)"
        title_r = "Normalized ratio"
    else:
        result = generate(0.0, A_VAC_TRUE, 0.0, seed=456)
        title_f = "Standard QFT (null)"
        title_r = "Normalized ratio"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_force_panel(axes[0], result, title_f)
    plot_ratio_panel(axes[1], result, title_r)
    save_figure(fig, "casimir_ripple_sim.png")
    print(verdict(result, title_f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Casimir ripple simulation (prediction #3)")
    parser.add_argument(
        "--mode",
        choices=["compare", "supersolid", "null"],
        default="compare",
        help="compare: side-by-side (default); supersolid or null: single row",
    )
    args = parser.parse_args()
    if args.mode == "compare":
        run_compare()
    else:
        run_single(args.mode)


if __name__ == "__main__":
    main()
