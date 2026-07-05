"""
Prediction #7: Gradient-threshold supersolid transition (CH framework).

Observables turn on only when |∇ρ| exceeds critical gradient |∇ρ|_c:
  chi = smooth_step(|∇ρ| - |∇ρ|_c)
  O = O_max * chi     (Casimir ripple, visibility deficit, etc.)

Run modes:
  python gradient_threshold_sim.py           # side-by-side: supersolid vs null (default)
  python gradient_threshold_sim.py --mode supersolid
  python gradient_threshold_sim.py --mode null
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Critical gradient (illustrative units: relative density gradient per meter)
GRAD_SAMPLE = np.linspace(0, 2.5, 80)
GRAD_CRIT = 1.0
NOISE = 0.03

# CH phenomenology: ripple amplitude vs gradient
ALPHA_MAX = 0.12
VISIBILITY_DIP_MAX = 0.15


def smooth_step(x: np.ndarray, width: float = 0.08) -> np.ndarray:
    """Smooth Θ(x): 0 for x << 0, 1 for x >> 0."""
    return 0.5 * (1.0 + np.tanh(x / width))


def chi(grad: np.ndarray, grad_c: float = GRAD_CRIT) -> np.ndarray:
    return smooth_step(grad - grad_c)


def ripple_amplitude(grad: np.ndarray, alpha_max: float = ALPHA_MAX) -> np.ndarray:
    return alpha_max * chi(grad)


def visibility_dip(grad: np.ndarray, dip_max: float = VISIBILITY_DIP_MAX) -> np.ndarray:
    """Fractional visibility reduction from supersolid patch along path."""
    return dip_max * chi(grad)


def model_null(grad, level):
    return np.full_like(grad, level, dtype=float)


def model_threshold(grad, level, amp, grad_c, width=0.08):
    return level + amp * smooth_step(grad - grad_c, width=width)


def fit_pair(grad: np.ndarray, y_meas: np.ndarray, sigma: float, threshold: bool) -> dict:
    if threshold:
        popt, _ = curve_fit(
            model_threshold,
            grad,
            y_meas,
            p0=[0.0, 0.05, GRAD_CRIT],
            sigma=np.full_like(y_meas, sigma),
            absolute_sigma=True,
            bounds=([0, 0, 0.2], [0.05, 0.5, 2.0]),
        )
        y_fit = model_threshold(grad, *popt)
    else:
        popt, _ = curve_fit(
            model_null,
            grad,
            y_meas,
            p0=[0.05],
            sigma=np.full_like(y_meas, sigma),
            absolute_sigma=True,
        )
        y_fit = model_null(grad, *popt)

    chi2 = float(np.sum(((y_meas - y_fit) / sigma) ** 2))
    return {"popt": popt, "y_fit": y_fit, "chi2": chi2}


def generate_signal(seed: int, with_threshold: bool) -> dict:
    rng = np.random.default_rng(seed)
    grad = GRAD_SAMPLE

    if with_threshold:
        alpha_true = ripple_amplitude(grad)
        vis_dip = visibility_dip(grad)
    else:
        alpha_true = np.zeros_like(grad)
        vis_dip = np.zeros_like(grad)

    alpha_meas = alpha_true + rng.normal(0, NOISE, size=grad.shape)
    vis_meas = vis_dip + rng.normal(0, NOISE, size=grad.shape)

    fit_alpha_thr = fit_pair(grad, alpha_meas, NOISE, threshold=True)
    fit_alpha_null = fit_pair(grad, alpha_meas, NOISE, threshold=False)
    fit_vis_thr = fit_pair(grad, vis_meas, NOISE, threshold=True)
    fit_vis_null = fit_pair(grad, vis_meas, NOISE, threshold=False)

    return {
        "grad": grad,
        "alpha_true": alpha_true,
        "alpha_meas": alpha_meas,
        "vis_true": vis_dip,
        "vis_meas": vis_meas,
        "fit_alpha_thr": fit_alpha_thr,
        "fit_alpha_null": fit_alpha_null,
        "fit_vis_thr": fit_vis_thr,
        "fit_vis_null": fit_vis_null,
    }


def plot_panel(ax: plt.Axes, grad, y_meas, y_true, y_fit_thr, y_fit_null, ylabel, title):
    ax.errorbar(grad, y_meas, yerr=NOISE, fmt="o", ms=4, capsize=3, label="Synthetic data")
    ax.plot(grad, y_true, "k-", lw=2, label="True (CH model)")
    ax.plot(grad, y_fit_thr, "-", label="Fit: threshold model")
    ax.plot(grad, y_fit_null, "--", label="Fit: null (flat)")
    ax.axvline(GRAD_CRIT, color="gray", ls=":", lw=1, label=rf"$|\nabla\rho|_c$={GRAD_CRIT}")
    ax.set_xlabel(r"$|\nabla\rho|$ (arb. units)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def print_verdict(result: dict, label: str) -> None:
    da = result["fit_alpha_null"]["chi2"] - result["fit_alpha_thr"]["chi2"]
    dv = result["fit_vis_null"]["chi2"] - result["fit_vis_thr"]["chi2"]
    p = result["fit_alpha_thr"]["popt"]
    print(f"\n{label}")
    print(f"  Ripple fit:  α_amp={p[1]:.4f}, |∇ρ|_c={p[2]:.3f} (planted {GRAD_CRIT})")
    print(f"  χ² ripple: threshold {result['fit_alpha_thr']['chi2']:.1f} vs null {result['fit_alpha_null']['chi2']:.1f} (Δχ²={da:.1f})")
    print(f"  χ² visibility: threshold {result['fit_vis_thr']['chi2']:.1f} vs null {result['fit_vis_null']['chi2']:.1f} (Δχ²={dv:.1f})")
    if result["alpha_true"].max() > 0:
        print(f"  → Gradient-gated supersolid signature (Δχ² > 0)")
    else:
        print(f"  → Consistent with uniform vacuum (no threshold signature)")


def run_compare() -> None:
    supersolid = generate_signal(seed=11, with_threshold=True)
    null = generate_signal(seed=22, with_threshold=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    g = supersolid["grad"]

    plot_panel(
        axes[0, 0],
        g,
        supersolid["alpha_meas"],
        supersolid["alpha_true"],
        supersolid["fit_alpha_thr"]["y_fit"],
        supersolid["fit_alpha_null"]["y_fit"],
        r"Casimir ripple amplitude $\alpha$",
        "Supersolid (hypothetical): ripple vs $|\\nabla\\rho|$",
    )
    plot_panel(
        axes[0, 1],
        null["grad"],
        null["alpha_meas"],
        null["alpha_true"],
        null["fit_alpha_thr"]["y_fit"],
        null["fit_alpha_null"]["y_fit"],
        r"Casimir ripple amplitude $\alpha$",
        "Standard QFT (null): flat vs $|\\nabla\\rho|$",
    )
    plot_panel(
        axes[1, 0],
        g,
        supersolid["vis_meas"],
        supersolid["vis_true"],
        supersolid["fit_vis_thr"]["y_fit"],
        supersolid["fit_vis_null"]["y_fit"],
        r"Visibility dip $\Delta V$",
        "Supersolid: interferometer visibility dip",
    )
    plot_panel(
        axes[1, 1],
        null["grad"],
        null["vis_meas"],
        null["vis_true"],
        null["fit_vis_thr"]["y_fit"],
        null["fit_vis_null"]["y_fit"],
        r"Visibility dip $\Delta V$",
        "Null: no visibility gradient effect",
    )

    fig.suptitle(
        "Prediction #7: CH gradient threshold — observables vs $|\\nabla\\rho|$",
        fontsize=12,
        y=1.01,
    )
    out = OUTPUT / "gradient_threshold_sim.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    print_verdict(supersolid, "Supersolid vacuum (gradient-gated)")
    print_verdict(null, "Standard QFT (uniform ρ)")


def run_single(mode: str) -> None:
    result = generate_signal(seed=11 if mode == "supersolid" else 22, with_threshold=(mode == "supersolid"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    g = result["grad"]
    title = "Supersolid (gradient-gated)" if mode == "supersolid" else "Standard QFT (null)"
    plot_panel(
        axes[0], g, result["alpha_meas"], result["alpha_true"],
        result["fit_alpha_thr"]["y_fit"], result["fit_alpha_null"]["y_fit"],
        r"Ripple $\alpha$", title,
    )
    plot_panel(
        axes[1], g, result["vis_meas"], result["vis_true"],
        result["fit_vis_thr"]["y_fit"], result["fit_vis_null"]["y_fit"],
        r"Visibility dip", title,
    )
    out = OUTPUT / "gradient_threshold_sim.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    print_verdict(result, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="CH gradient threshold simulation (prediction #7)")
    parser.add_argument(
        "--mode",
        choices=["compare", "supersolid", "null"],
        default="compare",
    )
    args = parser.parse_args()
    if args.mode == "compare":
        run_compare()
    else:
        run_single(args.mode)


if __name__ == "__main__":
    main()
