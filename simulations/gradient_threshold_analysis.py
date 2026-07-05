"""
Prediction #7 — gradient threshold analysis on real or synthetic (k, O, σ) data.

Fits flat (QFT) vs threshold (CH) models; optional joint fit across Casimir (#3)
and Mach–Zehnder (#6) channels with a shared k_c.

Usage:
  python gradient_threshold_analysis.py --demo
  python gradient_threshold_analysis.py --csv-alpha data/knob_alpha.csv --csv-vis data/knob_vis.csv
  python gradient_threshold_analysis.py --demo --joint

CSV format (header required):
  k,O,sigma
  0.0,0.002,0.01
  ...

k is the experimental knob in arbitrary or physical units (gap, curvature, etc.).
O is the observable (ripple amplitude α or visibility dip ΔV).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUTPUT = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data" / "gradient_threshold"
OUTPUT.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

DETECTION_DCHI2 = 9.0  # ~3σ for 2 extra parameters (amp, k_c)
SMOOTH_WIDTH_FRAC = 0.08  # smooth-step width as fraction of k range


def smooth_step(x: np.ndarray, width: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(x / max(width, 1e-30)))


def model_null(k: np.ndarray, level: float) -> np.ndarray:
    return np.full_like(k, level, dtype=float)


def model_threshold(
    k: np.ndarray, level: float, amp: float, k_c: float, width: float
) -> np.ndarray:
    return level + amp * smooth_step(k - k_c, width=width)


@dataclass
class ChannelData:
    name: str
    k: np.ndarray
    o: np.ndarray
    sigma: np.ndarray
    ylabel: str


@dataclass
class FitResult:
    model: str
    popt: np.ndarray
    perr: np.ndarray | None
    chi2: float
    ndof: int
    y_fit: np.ndarray
    width: float


def load_csv(path: Path) -> ChannelData:
    k_list, o_list, s_list = [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header (need k,O,sigma)")
        for row in reader:
            k_list.append(float(row["k"]))
            o_list.append(float(row["O"]))
            s_list.append(float(row["sigma"]))
    if not k_list:
        raise ValueError(f"{path}: no data rows")
    order = np.argsort(k_list)
    k = np.asarray(k_list)[order]
    o = np.asarray(o_list)[order]
    s = np.asarray(s_list)[order]
    return ChannelData(name=path.stem, k=k, o=o, sigma=s, ylabel="O")


def default_width(k: np.ndarray) -> float:
    span = float(np.max(k) - np.min(k))
    return max(span * SMOOTH_WIDTH_FRAC, 1e-12)


def fit_channel(ch: ChannelData, threshold: bool) -> FitResult:
    w = default_width(ch.k)
    if threshold:
        p0 = [float(np.median(ch.o)), max(float(np.std(ch.o)), 0.01), float(np.median(ch.k))]
        bounds = (
            [-np.inf, 0.0, np.min(ch.k) - w],
            [np.inf, np.inf, np.max(ch.k) + w],
        )

        def model(k, level, amp, k_c):
            return model_threshold(k, level, amp, k_c, w)

        popt, pcov = curve_fit(
            model,
            ch.k,
            ch.o,
            p0=p0,
            sigma=ch.sigma,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=20_000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
        y_fit = model(ch.k, *popt)
        ndof = len(ch.k) - 3
    else:
        p0 = [float(np.median(ch.o))]

        def model(k, level):
            return model_null(k, level)

        popt, pcov = curve_fit(
            model,
            ch.k,
            ch.o,
            p0=p0,
            sigma=ch.sigma,
            absolute_sigma=True,
            maxfev=10_000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
        y_fit = model(ch.k, *popt)
        ndof = len(ch.k) - 1

    chi2 = float(np.sum(((ch.o - y_fit) / ch.sigma) ** 2))
    return FitResult(
        model="threshold" if threshold else "null",
        popt=popt,
        perr=perr,
        chi2=chi2,
        ndof=ndof,
        y_fit=y_fit,
        width=w,
    )


def fit_joint(channels: list[ChannelData]) -> tuple[FitResult, FitResult]:
    """Joint null vs shared-k_c threshold across channels."""
    k_all = np.concatenate([c.k for c in channels])
    w = default_width(k_all)

    # Null: one level per channel
    def joint_null(k, *levels):
        out = []
        for i, ch in enumerate(channels):
            out.append(np.full_like(ch.k, levels[i]))
        return np.concatenate(out)

    p0_null = [float(np.median(ch.o)) for ch in channels]
    y_concat = np.concatenate([ch.o for ch in channels])
    sig_concat = np.concatenate([ch.sigma for ch in channels])

    popt_n, pcov_n = curve_fit(
        lambda k, *p: joint_null(k, *p),
        k_all,
        y_concat,
        p0=p0_null,
        sigma=sig_concat,
        absolute_sigma=True,
        maxfev=20_000,
    )
    y_null = joint_null(k_all, *popt_n)
    chi2_n = float(np.sum(((y_concat - y_null) / sig_concat) ** 2))
    ndof_n = len(y_concat) - len(channels)

    # Threshold: level_i, amp_i per channel + shared k_c
    def joint_thr(k, *params):
        n = len(channels)
        levels = params[:n]
        amps = params[n : 2 * n]
        k_c = params[2 * n]
        out = []
        for i, ch in enumerate(channels):
            out.append(model_threshold(ch.k, levels[i], amps[i], k_c, w))
        return np.concatenate(out)

    p0_thr = p0_null + [max(float(np.std(ch.o)), 0.01) for ch in channels] + [float(np.median(k_all))]
    lower = [-np.inf] * len(channels) + [0.0] * len(channels) + [np.min(k_all) - w]
    upper = [np.inf] * (2 * len(channels) + 1)
    popt_t, pcov_t = curve_fit(
        joint_thr,
        k_all,
        y_concat,
        p0=p0_thr,
        sigma=sig_concat,
        absolute_sigma=True,
        bounds=(lower, upper),
        maxfev=40_000,
    )
    y_thr = joint_thr(k_all, *popt_t)
    chi2_t = float(np.sum(((y_concat - y_thr) / sig_concat) ** 2))
    ndof_t = len(y_concat) - (2 * len(channels) + 1)

    fit_null = FitResult("joint_null", popt_n, None, chi2_n, ndof_n, y_null, w)
    fit_thr = FitResult("joint_threshold", popt_t, None, chi2_t, ndof_t, y_thr, w)
    return fit_null, fit_thr


def write_demo_csvs() -> tuple[Path, Path]:
    """Synthetic threshold data for pipeline testing."""
    rng = np.random.default_rng(42)
    k = np.linspace(0.0, 2.5, 12)
    k_c = 1.0
    w = default_width(k)
    alpha_true = 0.12 * smooth_step(k - k_c, w)
    vis_true = 0.15 * smooth_step(k - k_c, w)
    alpha = alpha_true + rng.normal(0, 0.03, size=k.shape)
    vis = vis_true + rng.normal(0, 0.03, size=k.shape)
    sig = np.full_like(k, 0.03)

    alpha_path = DATA / "demo_knob_alpha.csv"
    vis_path = DATA / "demo_knob_vis.csv"
    for path, o in [(alpha_path, alpha), (vis_path, vis)]:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "O", "sigma"])
            for ki, oi, si in zip(k, o, sig):
                writer.writerow([f"{ki:.6f}", f"{oi:.6f}", f"{si:.6f}"])
    return alpha_path, vis_path


def print_single_verdict(ch: ChannelData, null: FitResult, thr: FitResult) -> None:
    dchi2 = null.chi2 - thr.chi2
    print(f"\n--- {ch.name} ---")
    print(f"  Null χ²      = {null.chi2:.2f}  (ndof={null.ndof})")
    print(f"  Threshold χ² = {thr.chi2:.2f}  (ndof={thr.ndof})")
    print(f"  Δχ²          = {dchi2:.2f}  (detection cut: {DETECTION_DCHI2:.0f})")
    if thr.perr is not None and len(thr.popt) >= 3:
        print(f"  Fit: O₀={thr.popt[0]:.4f}, O_max={thr.popt[1]:.4f}, k_c={thr.popt[2]:.4f}")
    if dchi2 >= DETECTION_DCHI2:
        print("  → Threshold model preferred (CH-like turn-on)")
    else:
        print("  → Null (flat) model preferred or inconclusive")


def plot_channels(
    channels: list[ChannelData],
    fits_null: list[FitResult],
    fits_thr: list[FitResult],
    out_path: Path,
    title: str,
) -> None:
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    k_line = np.linspace(min(c.k.min() for c in channels), max(c.k.max() for c in channels), 200)

    for ax, ch, fn, ft in zip(axes[0], channels, fits_null, fits_thr):
        ax.errorbar(ch.k, ch.o, yerr=ch.sigma, fmt="o", capsize=3, label="Data")
        if fn.model == "null":
            ax.plot(ch.k, fn.y_fit, "--", label=f"Null (χ²={fn.chi2:.1f})")
        if ft.model == "threshold":
            w = ft.width
            y_thr = model_threshold(ch.k, *ft.popt[:3], w) if len(ft.popt) == 3 else ft.y_fit
            ax.plot(ch.k, y_thr, "-", label=f"Threshold (χ²={ft.chi2:.1f})")
            k_c = ft.popt[2] if len(ft.popt) >= 3 else np.nan
            if np.isfinite(k_c):
                ax.axvline(k_c, color="gray", ls=":", lw=1, label=rf"$k_c$={k_c:.2f}")
        ax.set_xlabel("Knob k")
        ax.set_ylabel(ch.ylabel)
        ax.set_title(ch.name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_analysis(
    alpha_csv: Path | None,
    vis_csv: Path | None,
    joint: bool,
    demo: bool,
) -> None:
    if demo:
        alpha_csv, vis_csv = write_demo_csvs()
        print(f"Wrote demo CSVs:\n  {alpha_csv}\n  {vis_csv}")

    channels: list[ChannelData] = []
    if alpha_csv is not None:
        ch = load_csv(alpha_csv)
        ch.name = "Casimir ripple α"
        ch.ylabel = r"Ripple amplitude $\alpha$"
        channels.append(ch)
    if vis_csv is not None:
        ch = load_csv(vis_csv)
        ch.name = "MZ visibility dip"
        ch.ylabel = r"Visibility dip $\Delta V$"
        channels.append(ch)
    if not channels:
        raise SystemExit("Provide --csv-alpha and/or --csv-vis, or use --demo")

    fits_null = [fit_channel(ch, threshold=False) for ch in channels]
    fits_thr = [fit_channel(ch, threshold=True) for ch in channels]

    print("=" * 60)
    print("CH Gradient Threshold Analysis (Prediction #7)")
    print("=" * 60)
    for ch, fn, ft in zip(channels, fits_null, fits_thr):
        print_single_verdict(ch, fn, ft)

    if joint and len(channels) >= 2:
        try:
            jn, jt = fit_joint(channels)
        except (RuntimeError, ValueError) as exc:
            print(f"\n--- Joint fit failed ({exc}) — per-channel results above ---")
            jn = jt = None
        if jn is not None and jt is not None:
            dchi2 = jn.chi2 - jt.chi2
            k_c = jt.popt[-1]
            print("\n--- Joint fit (shared k_c) ---")
            print(f"  Null χ²      = {jn.chi2:.2f}")
            print(f"  Threshold χ² = {jt.chi2:.2f}")
            print(f"  Δχ²          = {dchi2:.2f}")
            print(f"  Shared k_c   = {k_c:.4f}")
            n = len(channels)
            for i, ch in enumerate(channels):
                print(f"  {ch.name}: O₀={jt.popt[i]:.4f}, O_max={jt.popt[n + i]:.4f}")
            if dchi2 >= DETECTION_DCHI2:
                print("  → Joint threshold preferred")
            else:
                print("  → Joint null preferred")

            k1 = fits_thr[0].popt[2] if len(fits_thr[0].popt) >= 3 else np.nan
            k2 = fits_thr[1].popt[2] if len(fits_thr[1].popt) >= 3 else np.nan
            if np.isfinite(k1) and np.isfinite(k2):
                ratio = max(k1, k2) / max(min(k1, k2), 1e-30)
                if ratio > 2.0:
                    print(f"  ⚠ Per-channel k_c differ by factor {ratio:.1f} — single |∇ρ|_c CH ruled out")

    plot_channels(
        channels,
        fits_null,
        fits_thr,
        OUTPUT / "gradient_threshold_analysis.png",
        "Prediction #7: flat (QFT) vs threshold (CH) fit",
    )

    report = OUTPUT / "gradient_threshold_analysis.txt"
    with report.open("w") as f:
        f.write("CH gradient threshold analysis\n")
        for ch, fn, ft in zip(channels, fits_null, fits_thr):
            f.write(f"\n{ch.name}\n")
            f.write(f"  null chi2={fn.chi2:.4f}\n")
            f.write(f"  threshold chi2={ft.chi2:.4f}\n")
            f.write(f"  delta_chi2={fn.chi2 - ft.chi2:.4f}\n")
    print(f"Saved {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CH gradient threshold data analysis (#7)")
    parser.add_argument("--demo", action="store_true", help="Generate and analyze demo CSVs")
    parser.add_argument("--csv-alpha", type=Path, help="Casimir ripple CSV (k,O,sigma)")
    parser.add_argument("--csv-vis", type=Path, help="Visibility dip CSV (k,O,sigma)")
    parser.add_argument("--joint", action="store_true", help="Joint fit with shared k_c")
    args = parser.parse_args()

    if not args.demo and args.csv_alpha is None and args.csv_vis is None:
        args.demo = True

    run_analysis(args.csv_alpha, args.csv_vis, args.joint, args.demo)


if __name__ == "__main__":
    main()
