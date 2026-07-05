"""
Lab protocol analysis: signal channels + control channel → CH pass/fail verdict.

Implements the decision rules in docs/ch-lab-protocol-checklist.md:
  - Signal channels (α, ΔV): threshold vs flat
  - Joint k_c across signal channels
  - Control channel: must prefer flat for CH confirmation
  - Overall: CONSISTENT WITH CH / RULED OUT / INCONCLUSIVE

Usage:
  python control_channel_analysis.py --demo
  python control_channel_analysis.py \\
    --csv-alpha data/gradient_threshold/lab/run001_alpha.csv \\
    --csv-vis data/gradient_threshold/lab/run001_vis.csv \\
    --csv-control data/gradient_threshold/lab/run001_control.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gradient_threshold_analysis import (
    DETECTION_DCHI2,
    ChannelData,
    FitResult,
    fit_channel,
    fit_joint,
    load_csv,
    model_threshold,
    plot_channels,
    smooth_step,
    write_demo_csvs,
)

OUTPUT = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data" / "gradient_threshold" / "lab_demo"
OUTPUT.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

KC_AGREE_FACTOR = 2.0


class Verdict(str, Enum):
    CONSISTENT = "CONSISTENT WITH GRADIENT-GATED CH"
    RULED_OUT = "CH GRADIENT SECTOR RULED OUT (in scanned range)"
    SYSTEMATIC = "LIKELY SYSTEMATIC — NOT CH (control mirrors signal)"
    KC_SPLIT = "SINGLE |∇ρ|_c EXCLUDED (signal k_c disagree)"
    INCONCLUSIVE = "INCONCLUSIVE — more data or control needed"


@dataclass
class ProtocolResult:
    verdict: Verdict
    reasons: list[str]
    signal_channels: list[ChannelData]
    control: ChannelData | None
    signal_null: list[FitResult]
    signal_thr: list[FitResult]
    control_null: FitResult | None
    control_thr: FitResult | None
    joint_null: FitResult | None
    joint_thr: FitResult | None


def write_lab_demo_csvs() -> tuple[Path, Path, Path]:
    """Signal channels with threshold; control channel flat (χ_wall-like)."""
    rng = np.random.default_rng(99)
    k = np.linspace(2.0, 20.0, 12)
    k_c = 10.0
    w = max((k.max() - k.min()) * 0.08, 1e-12)
    alpha_true = 0.10 * smooth_step(k - k_c, w)
    vis_true = 0.12 * smooth_step(k - k_c, w)
    control_true = np.full_like(k, 0.048)
    sig = np.full_like(k, 0.012)

    paths = []
    for name, o_true in [
        ("lab_demo_alpha.csv", alpha_true),
        ("lab_demo_vis.csv", vis_true),
        ("lab_demo_control.csv", control_true),
    ]:
        path = DATA / name
        o_meas = o_true + rng.normal(0, sig[0], size=k.shape)
        with path.open("w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["k", "O", "sigma"])
            for ki, oi in zip(k, o_meas):
                wtr.writerow([f"{ki:.6f}", f"{oi:.6f}", f"{sig[0]:.6f}"])
        paths.append(path)
    return paths[0], paths[1], paths[2]


def evaluate_protocol(
    signal_channels: list[ChannelData],
    control: ChannelData | None,
    kc_agree_factor: float = KC_AGREE_FACTOR,
) -> ProtocolResult:
    signal_null = [fit_channel(ch, False) for ch in signal_channels]
    signal_thr = [fit_channel(ch, True) for ch in signal_channels]

    joint_null = joint_thr = None
    if len(signal_channels) >= 2:
        try:
            joint_null, joint_thr = fit_joint(signal_channels)
        except (RuntimeError, ValueError):
            joint_null = joint_thr = None

    control_null = control_thr = None
    if control is not None:
        control_null = fit_channel(control, False)
        control_thr = fit_channel(control, True)

    reasons: list[str] = []
    verdict = Verdict.INCONCLUSIVE

    # Per-signal threshold detection
    signal_detect = []
    k_cs = []
    for ch, fn, ft in zip(signal_channels, signal_null, signal_thr):
        dchi2 = fn.chi2 - ft.chi2
        detected = dchi2 >= DETECTION_DCHI2
        signal_detect.append(detected)
        if len(ft.popt) >= 3:
            k_cs.append(float(ft.popt[2]))
        reasons.append(
            f"Signal '{ch.name}': Δχ²={dchi2:.2f} "
            f"({'threshold' if detected else 'null/inconclusive'})"
        )

    all_signal_flat = not any(signal_detect)
    any_signal_threshold = any(signal_detect)

    # k_c agreement
    kc_ok = True
    if len(k_cs) >= 2:
        ratio = max(k_cs) / max(min(k_cs), 1e-30)
        kc_ok = ratio <= kc_agree_factor
        reasons.append(f"Signal k_c values: {k_cs} (ratio={ratio:.2f}, need ≤{kc_agree_factor})")
        if not kc_ok and any_signal_threshold:
            return ProtocolResult(
                Verdict.KC_SPLIT, reasons + ["Per-channel k_c disagree."],
                signal_channels, control, signal_null, signal_thr,
                control_null, control_thr, joint_null, joint_thr,
            )

    # Control channel
    control_flat = True
    if control is not None and control_null is not None and control_thr is not None:
        dchi2_c = control_null.chi2 - control_thr.chi2
        control_flat = dchi2_c < DETECTION_DCHI2
        control_threshold = dchi2_c >= DETECTION_DCHI2
        reasons.append(
            f"Control '{control.name}': Δχ²={dchi2_c:.2f} "
            f"({'flat OK' if control_flat else 'THRESHOLD — systematic warning'})"
        )
        if control_threshold and any_signal_threshold:
            return ProtocolResult(
                Verdict.SYSTEMATIC,
                reasons + ["Control shows threshold like signal — likely artifact."],
                signal_channels, control, signal_null, signal_thr,
                control_null, control_thr, joint_null, joint_thr,
            )

    # Joint fit
    joint_ok = False
    if joint_null is not None and joint_thr is not None:
        dchi2_j = joint_null.chi2 - joint_thr.chi2
        joint_ok = dchi2_j >= DETECTION_DCHI2
        reasons.append(f"Joint signal fit: Δχ²={dchi2_j:.2f}")

    # Final verdict
    if all_signal_flat:
        verdict = Verdict.RULED_OUT
        reasons.append("All signal channels prefer flat — CH gradient turn-on not seen in range.")
    elif any_signal_threshold and control is not None and control_flat and kc_ok:
        if len(signal_channels) == 1 or joint_ok:
            verdict = Verdict.CONSISTENT
            reasons.append("Signal threshold + flat control + k_c agreement.")
        else:
            reasons.append("Threshold in some channels but joint fit weak — inconclusive.")
    elif any_signal_threshold and control is None:
        reasons.append("Threshold in signal but no control channel — cannot confirm CH.")
    elif any_signal_threshold and not kc_ok:
        verdict = Verdict.KC_SPLIT
    else:
        reasons.append("Mixed or weak significance — collect more data.")

    return ProtocolResult(
        verdict, reasons, signal_channels, control,
        signal_null, signal_thr, control_null, control_thr, joint_null, joint_thr,
    )


def plot_protocol(result: ProtocolResult, out_path: Path) -> None:
    panels = list(result.signal_channels)
    labels_null = list(result.signal_null)
    labels_thr = list(result.signal_thr)
    if result.control is not None and result.control_null and result.control_thr:
        panels.append(result.control)
        labels_null.append(result.control_null)
        labels_thr.append(result.control_thr)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    for ax, ch, fn, ft in zip(axes[0], panels, labels_null, labels_thr):
        ax.errorbar(ch.k, ch.o, yerr=ch.sigma, fmt="o", capsize=3, label="Data")
        ax.plot(ch.k, fn.y_fit, "--", label=f"Null (χ²={fn.chi2:.1f})")
        w = ft.width
        if len(ft.popt) >= 3:
            y_thr = model_threshold(ch.k, *ft.popt[:3], w)
            ax.plot(ch.k, y_thr, "-", label=f"Threshold (χ²={ft.chi2:.1f})")
            ax.axvline(ft.popt[2], color="gray", ls=":", lw=1)
        ax.set_xlabel("Knob k")
        ax.set_ylabel(ch.ylabel)
        ax.set_title(ch.name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"CH lab protocol — verdict: {result.verdict.value}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def print_report(result: ProtocolResult, out_path: Path) -> None:
    print("\n" + "=" * 60)
    print("CH LAB PROTOCOL VERDICT")
    print("=" * 60)
    for line in result.reasons:
        print(f"  {line}")
    print(f"\n>>> {result.verdict.value} <<<\n")

    with out_path.open("w") as f:
        f.write("CH lab protocol analysis\n")
        f.write(f"verdict: {result.verdict.value}\n\n")
        for line in result.reasons:
            f.write(f"{line}\n")
    print(f"Saved {out_path}")


def run(
    alpha_csv: Path | None,
    vis_csv: Path | None,
    control_csv: Path | None,
    demo: bool,
) -> ProtocolResult:
    if demo:
        alpha_csv, vis_csv, control_csv = write_lab_demo_csvs()
        print(f"Wrote lab demo CSVs in {DATA}/")

    signals: list[ChannelData] = []
    if alpha_csv is not None:
        ch = load_csv(alpha_csv)
        ch.name = "Signal: Casimir α"
        ch.ylabel = r"$\alpha$"
        signals.append(ch)
    if vis_csv is not None:
        ch = load_csv(vis_csv)
        ch.name = "Signal: MZ ΔV"
        ch.ylabel = r"$\Delta V$"
        signals.append(ch)
    if not signals:
        raise SystemExit("Provide at least one signal CSV (--csv-alpha and/or --csv-vis)")

    control = None
    if control_csv is not None:
        control = load_csv(control_csv)
        control.name = "Control (expect flat)"
        control.ylabel = "Control O"

    result = evaluate_protocol(signals, control)
    plot_protocol(result, OUTPUT / "control_channel_analysis.png")
    print_report(result, OUTPUT / "control_channel_analysis.txt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CH lab protocol: signal + control verdict")
    parser.add_argument("--demo", action="store_true", help="Synthetic signal + flat control")
    parser.add_argument("--csv-alpha", type=Path, help="Signal: Casimir ripple CSV")
    parser.add_argument("--csv-vis", type=Path, help="Signal: visibility dip CSV")
    parser.add_argument("--csv-control", type=Path, help="Control channel CSV (expect flat)")
    args = parser.parse_args()

    if not args.demo and args.csv_alpha is None and args.csv_vis is None:
        args.demo = True

    run(args.csv_alpha, args.csv_vis, args.csv_control, args.demo)


if __name__ == "__main__":
    main()
