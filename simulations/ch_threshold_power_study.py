#!/usr/bin/env python3
"""
Prediction #7 — Monte Carlo power study for threshold detection.

Given GPE-predicted chi vs gap d (wall-coupled ripple channel), estimates:
  - detection power vs alpha_max, n_k gap points, repeats per k, and sigma
  - minimum alpha_max for 90% power at realistic AFM ripple uncertainty
  - recommended gap scan grid

Usage:
  python ch_threshold_power_study.py
  python ch_threshold_power_study.py --quick
  python ch_threshold_power_study.py --sigma 0.01 --target-power 0.9
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_to_threshold_demo import gpe_observables_vs_gap
from gradient_threshold_analysis import DETECTION_DCHI2, ChannelData, fit_channel

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# AFM ripple uncertainty presets (absolute sigma on extracted alpha)
SIGMA_PRESETS = {
    "optimistic": 0.005,
    "moderate": 0.010,
    "conservative": 0.020,
}


@dataclass
class PowerStudyConfig:
    xi_m: float = 50e-9
    d_min_m: float = 40e-9
    d_max_m: float = 600e-9
    n_k: int = 12
    n_repeat: int = 20
    sigma_alpha: float = 0.01
    alpha_max: float = 0.12
    coupling: str = "mid"
    n_trials: int = 400
    target_power: float = 0.90
    seed: int = 42


@dataclass
class PowerResult:
    power: float
    median_dchi2: float
    frac_dchi2_above_cut: float


def gpe_alpha_curve(cfg: PowerStudyConfig, alpha_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k, alpha_true, d_m) from GPE wall-coupled mapping."""
    ch = CHParams(xi=cfg.xi_m)
    d_array = np.logspace(np.log10(cfg.d_min_m), np.log10(cfg.d_max_m), cfg.n_k)
    data = gpe_observables_vs_gap(ch, d_array, alpha_max, vis_max=0.15, coupling=cfg.coupling)
    return data["k"], data["alpha"], data["d_m"]


def trial_detected(
    k: np.ndarray,
    alpha_true: np.ndarray,
    sigma_single: float,
    n_repeat: int,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """One MC trial: average n_repeat shots per k, fit flat vs threshold."""
    sigma_eff = sigma_single / np.sqrt(max(n_repeat, 1))
    o_meas = alpha_true + rng.normal(0.0, sigma_eff, size=k.shape)
    ch = ChannelData(
        name="mc_trial",
        k=k,
        o=o_meas,
        sigma=np.full_like(k, sigma_eff),
        ylabel="alpha",
    )
    fit_null = fit_channel(ch, threshold=False)
    fit_thr = fit_channel(ch, threshold=True)
    dchi2 = fit_null.chi2 - fit_thr.chi2
    return dchi2 >= DETECTION_DCHI2, float(dchi2)


def estimate_power(
    k: np.ndarray,
    alpha_true: np.ndarray,
    sigma_single: float,
    n_repeat: int,
    n_trials: int,
    seed: int,
) -> PowerResult:
    rng = np.random.default_rng(seed)
    hits = 0
    dchi2_list: list[float] = []
    for t in range(n_trials):
        detected, dchi2 = trial_detected(k, alpha_true, sigma_single, n_repeat, rng)
        hits += int(detected)
        dchi2_list.append(dchi2)
    dchi2_arr = np.asarray(dchi2_list)
    return PowerResult(
        power=hits / n_trials,
        median_dchi2=float(np.median(dchi2_arr)),
        frac_dchi2_above_cut=float(np.mean(dchi2_arr >= DETECTION_DCHI2)),
    )


def bracket_alpha_for_power(
    cfg: PowerStudyConfig,
    alpha_lo: float = 0.005,
    alpha_hi: float = 0.30,
    tol: float = 0.005,
) -> tuple[float, PowerResult]:
    """Binary search smallest alpha_max with power >= target."""
    k_ref, _, _ = gpe_alpha_curve(cfg, alpha_max=1.0)  # shape only

    def power_at(alpha: float) -> PowerResult:
        _, alpha_true, _ = gpe_alpha_curve(cfg, alpha_max=alpha)
        return estimate_power(
            k_ref,
            alpha_true,
            cfg.sigma_alpha,
            cfg.n_repeat,
            cfg.n_trials,
            cfg.seed + int(alpha * 1e4),
        )

    lo, hi = alpha_lo, alpha_hi
    res_hi = power_at(hi)
    while res_hi.power < cfg.target_power and hi < 1.0:
        hi = min(hi * 1.5, 1.0)
        res_hi = power_at(hi)
    if res_hi.power < cfg.target_power:
        return float("nan"), res_hi

    res_lo = power_at(lo)
    best_alpha = hi
    best_res = res_hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        res_mid = power_at(mid)
        if res_mid.power >= cfg.target_power:
            hi = mid
            best_alpha = mid
            best_res = res_mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return best_alpha, best_res


def sweep_n_k(
    cfg: PowerStudyConfig,
    n_k_values: list[int],
) -> list[tuple[int, PowerResult]]:
    out: list[tuple[int, PowerResult]] = []
    for n_k in n_k_values:
        sub = PowerStudyConfig(**{**cfg.__dict__, "n_k": n_k})
        k, alpha_true, _ = gpe_alpha_curve(sub, sub.alpha_max)
        res = estimate_power(
            k, alpha_true, sub.sigma_alpha, sub.n_repeat, sub.n_trials, sub.seed + n_k
        )
        out.append((n_k, res))
    return out


def format_gap_table(k: np.ndarray, d_m: np.ndarray, alpha_true: np.ndarray) -> list[str]:
    lines = ["Recommended gap scan (GPE bulk alpha signal channel):", "  d [nm]    k [1/nm]   alpha_eff"]
    for d, ki, a in zip(d_m, k, alpha_true):
        k_per_nm = 1.0 / (d * 1e9)
        lines.append(f"  {d*1e9:7.1f}   {k_per_nm:8.4f}   {a:.5f}")
    return lines


def plot_power_curves(
    cfg: PowerStudyConfig,
    alpha_grid: np.ndarray,
    powers: np.ndarray,
    n_k_sweep: list[tuple[int, PowerResult]],
    alpha_90: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(alpha_grid * 100, powers * 100, "b-o", ms=4)
    axes[0].axhline(cfg.target_power * 100, color="k", ls="--", lw=0.8, label=f"{cfg.target_power:.0%} power")
    if np.isfinite(alpha_90):
        axes[0].axvline(alpha_90 * 100, color="r", ls=":", label=rf"$\alpha_{{\max}}$ @ 90% = {alpha_90*100:.1f}%")
    axes[0].set_xlabel(r"$\alpha_{\max}$ [%]")
    axes[0].set_ylabel("Detection power [%]")
    axes[0].set_title(
        rf"Power vs $\alpha_{{\max}}$ ($n_k$={cfg.n_k}, $\sigma$={cfg.sigma_alpha:.3f}, "
        rf"$N$={cfg.n_repeat}/point)"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 102)

    nk = [n for n, _ in n_k_sweep]
    pw = [r.power * 100 for _, r in n_k_sweep]
    axes[1].plot(nk, pw, "g-s", ms=5)
    axes[1].axhline(cfg.target_power * 100, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel(r"Number of gap settings $n_k$")
    axes[1].set_ylabel("Detection power [%]")
    axes[1].set_title(rf"Power vs $n_k$ ($\alpha_{{\max}}$={cfg.alpha_max:.2f})")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 102)

    fig.suptitle(
        rf"CH threshold power study ($\xi$={cfg.xi_m*1e9:.0f} nm, {cfg.coupling} coupling)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_study(cfg: PowerStudyConfig, quick: bool = False) -> str:
    if quick:
        cfg = PowerStudyConfig(**{**cfg.__dict__, "n_trials": min(cfg.n_trials, 80)})

    k, alpha_true, d_m = gpe_alpha_curve(cfg, cfg.alpha_max)
    baseline = estimate_power(
        k, alpha_true, cfg.sigma_alpha, cfg.n_repeat, cfg.n_trials, cfg.seed
    )

    alpha_grid = np.linspace(0.02, 0.20, 10)
    powers = []
    for i, a in enumerate(alpha_grid):
        _, a_true, _ = gpe_alpha_curve(cfg, a)
        res = estimate_power(
            k, a_true, cfg.sigma_alpha, cfg.n_repeat, cfg.n_trials, cfg.seed + i + 100
        )
        powers.append(res.power)
    powers_arr = np.asarray(powers)

    alpha_90, res_90 = bracket_alpha_for_power(cfg)
    n_k_sweep = sweep_n_k(cfg, [6, 8, 10, 12, 14, 16, 20])

    lines = [
        "=" * 72,
        "CH THRESHOLD POWER STUDY (Prediction #7, Tier A Casimir alpha)",
        "=" * 72,
        f"xi = {cfg.xi_m*1e9:.1f} nm",
        f"Gap range: {cfg.d_min_m*1e9:.0f}–{cfg.d_max_m*1e9:.0f} nm",
        f"Coupling: {cfg.coupling} (bulk chi for gap-d signal; wall for controls)",
        f"sigma_alpha (per shot) = {cfg.sigma_alpha:.4f}",
        f"Repeats per k = {cfg.n_repeat}  =>  sigma_eff = {cfg.sigma_alpha/np.sqrt(cfg.n_repeat):.4f}",
        f"n_k = {cfg.n_k},  n_MC trials = {cfg.n_trials}",
        f"Detection cut: delta_chi2 >= {DETECTION_DCHI2}",
        "",
        f"At alpha_max = {cfg.alpha_max:.3f}:",
        f"  power = {baseline.power:.1%}",
        f"  median delta_chi2 = {baseline.median_dchi2:.1f}",
        "",
    ]

    if np.isfinite(alpha_90):
        lines += [
            f"Minimum alpha_max for {cfg.target_power:.0%} power:",
            f"  alpha_max >= {alpha_90:.4f}  ({alpha_90*100:.2f}% ripple ceiling)",
            f"  (median delta_chi2 at bracket = {res_90.median_dchi2:.1f})",
            "",
        ]
    else:
        lines += [
            f"Could not reach {cfg.target_power:.0%} power with alpha_max <= 1.0",
            "  => effect may be below detectability at this sigma / n_k",
            "",
        ]

    lines.append("Power vs n_k (fixed alpha_max):")
    for n_k, res in n_k_sweep:
        mark = " <--" if res.power >= cfg.target_power else ""
        lines.append(f"  n_k={n_k:2d}: power={res.power:.1%}{mark}")
    lines.append("")

    lines.append("Sigma presets (absolute uncertainty on extracted alpha):")
    for name, sig in SIGMA_PRESETS.items():
        sub = PowerStudyConfig(**{**cfg.__dict__, "sigma_alpha": sig})
        a_req, _ = bracket_alpha_for_power(sub)
        if np.isfinite(a_req):
            lines.append(f"  {name:12s} sigma={sig:.3f}  =>  alpha_max >= {a_req:.4f} for 90% power")
        else:
            lines.append(f"  {name:12s} sigma={sig:.3f}  =>  not reachable at alpha_max <= 1")
    lines.append("")

    lines.extend(format_gap_table(k, d_m, alpha_true))
    lines.append("")
    lines.append(
        "Interpretation: if GPE bulk chi tracks the real AFM ripple-vs-gap channel, "
        "these are the minimum alpha_max and scan density needed for 90% chance "
        "that the protocol declares a threshold (delta_chi2 >= 9) before unblinding."
    )

    report = "\n".join(lines) + "\n"
    plot_power_curves(cfg, alpha_grid, powers_arr, n_k_sweep, alpha_90, OUTPUT / "ch_threshold_power_study.png")
    (OUTPUT / "ch_threshold_power_study.txt").write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo power for CH threshold detection")
    parser.add_argument("--xi", type=float, default=50e-9, help="Healing length [m]")
    parser.add_argument("--d-min", type=float, default=40e-9)
    parser.add_argument("--d-max", type=float, default=600e-9)
    parser.add_argument("--n-k", type=int, default=12, help="Gap settings in scan")
    parser.add_argument("--n-repeat", type=int, default=20, help="Repeats averaged per k")
    parser.add_argument("--sigma", type=float, default=0.01, help="Per-shot sigma on alpha")
    parser.add_argument(
        "--sigma-preset",
        choices=list(SIGMA_PRESETS),
        default=None,
        help="Override --sigma with optimistic/moderate/conservative",
    )
    parser.add_argument("--alpha-max", type=float, default=0.12)
    parser.add_argument("--coupling", choices=["mid", "wall", "mixed"], default="mid")
    parser.add_argument("--n-trials", type=int, default=400, help="Monte Carlo trials")
    parser.add_argument("--target-power", type=float, default=0.90)
    parser.add_argument("--quick", action="store_true", help="Fewer MC trials (faster)")
    args = parser.parse_args()

    sigma = SIGMA_PRESETS[args.sigma_preset] if args.sigma_preset else args.sigma
    cfg = PowerStudyConfig(
        xi_m=args.xi,
        d_min_m=args.d_min,
        d_max_m=args.d_max,
        n_k=args.n_k,
        n_repeat=args.n_repeat,
        sigma_alpha=sigma,
        alpha_max=args.alpha_max,
        coupling=args.coupling,
        n_trials=args.n_trials,
        target_power=args.target_power,
    )

    report = run_study(cfg, quick=args.quick)
    print(report)
    print(f"Wrote {OUTPUT / 'ch_threshold_power_study.png'}")
    print(f"Wrote {OUTPUT / 'ch_threshold_power_study.txt'}")


if __name__ == "__main__":
    main()
