"""
Lab → ξ → gravity bridge for Chronos-Hydrodynamics.

Reads Prediction #7 protocol outputs (or CSVs), extracts k_c with
k = 1/d_nm [1/nm], maps to ξ via gap-turn-on hypothesis, cross-checks
Casimir-lattice ξ if a_vac is supplied, and forecasts gravity v3 with
self-consistent α_G calibration.

Usage:
  python ch_xi_lab_bridge.py --demo
  python ch_xi_lab_bridge.py \\
    --csv-alpha data/gradient_threshold/lab/run001_alpha.csv \\
    --csv-vis data/gradient_threshold/lab/run001_vis.csv \\
    --a-vac 150e-9
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from ch_dispersion_core import CHParams
from ch_gpe_gravity import (
    AlphaGCalibration,
    alpha_g_profile_corrected,
    calibrate_alpha_g_from_defect,
    solve_gravity_sm_v3,
)
from ch_xi_prediction import (
    XiConsistencyReport,
    XiPrediction,
    consistency_report,
    predict_xi_from_casimir_lattice,
    predict_xi_from_kc,
)
from control_channel_analysis import (
    evaluate_protocol,
    write_lab_demo_csvs,
)
from gradient_threshold_analysis import ChannelData, fit_channel, fit_joint, load_csv

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


@dataclass
class KcExtraction:
    """Threshold knob k_c with k = 1/d_nm."""

    k_c_per_nm: float
    d_turnon_m: float
    per_channel: dict[str, float]
    joint_k_c: float | None
    method: str

    @staticmethod
    def from_kc(k_c: float, method: str, per_channel: dict[str, float], joint: float | None) -> KcExtraction:
        return KcExtraction(
            k_c_per_nm=k_c,
            d_turnon_m=1.0 / (k_c * 1e9),
            per_channel=per_channel,
            joint_k_c=joint,
            method=method,
        )


def kc_from_threshold_fit(fit, *, joint: bool = False) -> float | None:
    if fit is None:
        return None
    if joint or getattr(fit, "model", "") == "joint_threshold":
        if len(fit.popt) < 1:
            return None
        k_c = float(fit.popt[-1])
    elif len(fit.popt) < 3:
        return None
    else:
        k_c = float(fit.popt[2])
    return k_c if np.isfinite(k_c) and k_c > 0 else None


def extract_kc_from_channels(channels: list[ChannelData]) -> KcExtraction:
    """Fit threshold model per channel; prefer joint k_c when ≥2 channels."""
    per_channel: dict[str, float] = {}
    k_cs: list[float] = []
    for ch in channels:
        thr = fit_channel(ch, threshold=True)
        k_c = kc_from_threshold_fit(thr)
        if k_c is not None:
            per_channel[ch.name] = k_c
            k_cs.append(k_c)

    joint_k: float | None = None
    if len(channels) >= 2:
        try:
            _, joint_thr = fit_joint(channels)
            joint_k = kc_from_threshold_fit(joint_thr, joint=True)
        except (RuntimeError, ValueError):
            joint_k = None

    if joint_k is not None:
        return KcExtraction.from_kc(joint_k, "joint", per_channel, joint_k)
    if k_cs:
        k_med = float(median(k_cs))
        return KcExtraction.from_kc(k_med, "median", per_channel, None)
    raise ValueError("Could not extract k_c from channels (threshold fits failed)")


def extract_kc_from_protocol(result) -> KcExtraction:
    """Use joint fit from protocol when available, else per-signal median."""
    per_channel: dict[str, float] = {}
    for ch, ft in zip(result.signal_channels, result.signal_thr):
        k_c = kc_from_threshold_fit(ft)
        if k_c is not None:
            per_channel[ch.name] = k_c

    joint_k = kc_from_threshold_fit(result.joint_thr, joint=True)
    if joint_k is not None:
        return KcExtraction.from_kc(joint_k, "protocol_joint", per_channel, joint_k)

    k_cs = list(per_channel.values())
    if not k_cs:
        raise ValueError("Protocol has no threshold k_c in signal channels")
    return KcExtraction.from_kc(float(median(k_cs)), "protocol_median", per_channel, None)


@dataclass
class LabXiBridgeResult:
    kc: KcExtraction
    xi_gap: XiPrediction
    xi_casimir: XiPrediction | None
    reports: list[XiConsistencyReport]
    consensus_xi_m: float
    consensus_label: str


def build_xi_bridge(
    kc: KcExtraction,
    a_vac_m: float | None = None,
    alpha_g: float = 1.0,
) -> LabXiBridgeResult:
    xi_gap = predict_xi_from_kc(kc.k_c_per_nm, alpha_g=alpha_g)
    xi_casimir = predict_xi_from_casimir_lattice(a_vac_m, alpha_g=alpha_g) if a_vac_m else None

    reports = [consistency_report(xi_gap)]
    if xi_casimir is not None:
        reports.append(consistency_report(xi_casimir))

    if xi_casimir is not None:
        ratio = xi_gap.xi_m / xi_casimir.xi_m
        if 0.5 <= ratio <= 2.0:
            consensus = float(np.sqrt(xi_gap.xi_m * xi_casimir.xi_m))
            label = "geometric_mean(gap_turnon, casimir_lattice)"
        else:
            consensus = xi_gap.xi_m
            label = "gap_turnon (Casimir lattice disagrees >2×)"
    else:
        consensus = xi_gap.xi_m
        label = "gap_turnon from k_c"

    return LabXiBridgeResult(
        kc=kc,
        xi_gap=xi_gap,
        xi_casimir=xi_casimir,
        reports=reports,
        consensus_xi_m=consensus,
        consensus_label=label,
    )


@dataclass
class LabGravityForecast:
    xi_m: float
    ch: CHParams
    gravity_assumed: object
    alpha_calibration: AlphaGCalibration
    gravity_calibrated: object
    alpha_g_profile: float
    f_profile: float
    gravity_profile: object


def forecast_gravity_from_xi(
    xi_m: float,
    alpha_g_assumed: float = 1.0,
    r_s_hat: float = 1.0,
) -> LabGravityForecast:
    ch = CHParams(xi=xi_m, alpha_g=alpha_g_assumed)
    gravity = solve_gravity_sm_v3(ch, r_s_hat=r_s_hat)
    cal, gravity_cal = calibrate_alpha_g_from_defect(gravity)
    alpha_prof, f_prof, gravity_prof = alpha_g_profile_corrected(gravity)
    return LabGravityForecast(
        xi_m=xi_m,
        ch=ch,
        gravity_assumed=gravity,
        alpha_calibration=cal,
        gravity_calibrated=gravity_cal,
        alpha_g_profile=float(alpha_prof),
        f_profile=float(f_prof),
        gravity_profile=gravity_prof,
    )


def format_kc_extraction(kc: KcExtraction) -> list[str]:
    lines = [
        f"k_c extraction ({kc.method})",
        f"  k_c = {kc.k_c_per_nm:.4f} 1/nm",
        f"  d_c = {kc.d_turnon_m:.4e} m  ({kc.d_turnon_m*1e9:.2f} nm)",
    ]
    if kc.per_channel:
        for name, val in kc.per_channel.items():
            lines.append(f"  per-channel {name}: k_c={val:.4f} 1/nm")
    if kc.joint_k_c is not None:
        lines.append(f"  joint k_c: {kc.joint_k_c:.4f} 1/nm")
    return lines


def format_xi_bridge(bridge: LabXiBridgeResult) -> list[str]:
    lines = format_kc_extraction(bridge.kc)
    lines.append("")
    lines.append(f"Consensus ξ ({bridge.consensus_label}): {bridge.consensus_xi_m:.4e} m")
    for rep in bridge.reports:
        lines.append("")
        lines.extend(rep.summary_lines())
    return lines


def format_alpha_g_calibration(cal: AlphaGCalibration) -> list[str]:
    return [
        "α_G calibration (defect profile → Newton factor ≈ 1)",
        f"  α_G assumed:              {cal.alpha_g_assumed:.4e}",
        f"  α_G calibrated:           {cal.alpha_g_calibrated:.4e}",
        f"  α_G hydro-only reference:   {cal.alpha_g_hydro_only_reference:.4e}",
        f"  Newton @ assumed α_G:       {cal.newton_slope_at_assumed:.4e}",
        f"  Newton Q contribution:      {cal.newton_slope_q:.4e}",
        f"  Newton hydro per unit α_G:  {cal.newton_slope_hydro_per_unit_alpha:.4e}",
        f"  Newton after calibration:   {cal.newton_slope_after_calibration:.4e}",
        f"  G implied (slope × G_meas): {cal.g_implied_after_calibration:.4e}",
        f"  Mass deficit (profile):     {cal.mass_deficit_kg:.4e} kg",
    ]


def format_gravity_forecast(fc: LabGravityForecast) -> list[str]:
    g = fc.gravity_profile
    lines = [
        "",
        f"Gravity v3 forecast @ ξ = {fc.xi_m:.4e} m",
        f"  solver: {g.solver}",
        f"  Newton total (α_G=1):        {fc.gravity_assumed.newton_slope:.4e}",
        f"  Newton total (profile rule): {g.newton_slope:.4e}",
        f"  Newton total (slope cal):    {fc.gravity_calibrated.newton_slope:.4e}",
        "α_G profile rule (f = 1/N_hydro@grain ref)",
        f"  α_G hydro-only reference:   {fc.alpha_calibration.alpha_g_hydro_only_reference:.4e}",
        f"  f_profile:                  {fc.f_profile:.4e}",
        f"  α_G profile:                {fc.alpha_g_profile:.4e}",
    ]
    lines.extend(format_alpha_g_calibration(fc.alpha_calibration))
    return lines


def run_demo_pipeline(a_vac_m: float = 150e-9, r_s_hat: float = 1.0) -> str:
    alpha_csv, vis_csv, control_csv = write_lab_demo_csvs()
    alpha_ch = load_csv(alpha_csv)
    vis_ch = load_csv(vis_csv)
    control_ch = load_csv(control_csv)
    protocol = evaluate_protocol([alpha_ch, vis_ch], control_ch)
    kc = extract_kc_from_protocol(protocol)
    bridge = build_xi_bridge(kc, a_vac_m=a_vac_m)
    forecast = forecast_gravity_from_xi(bridge.consensus_xi_m, r_s_hat=r_s_hat)

    lines = [
        "=" * 72,
        "CH LAB → ξ → GRAVITY BRIDGE (demo)",
        "=" * 72,
        f"Protocol verdict: {protocol.verdict.value}",
        "",
    ]
    lines.extend(format_xi_bridge(bridge))
    lines.extend(format_gravity_forecast(forecast))

    report = "\n".join(lines) + "\n"
    out_path = OUTPUT / "ch_lab_xi_bridge_report.txt"
    out_path.write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab k_c → ξ → gravity v3 bridge")
    parser.add_argument("--demo", action="store_true", help="Run lab demo CSVs end-to-end")
    parser.add_argument("--csv-alpha", type=Path, default=None)
    parser.add_argument("--csv-vis", type=Path, default=None)
    parser.add_argument("--csv-control", type=Path, default=None)
    parser.add_argument("--a-vac", type=float, default=150e-9, help="Casimir ripple period [m]")
    parser.add_argument("--rs-hat", type=float, default=1.0, help="Schwarzschild radius in ξ units")
    parser.add_argument("--skip-gravity", action="store_true")
    args = parser.parse_args()

    if args.demo:
        report = run_demo_pipeline(a_vac_m=args.a_vac, r_s_hat=args.rs_hat)
        print(report)
        print(f"Wrote {OUTPUT / 'ch_lab_xi_bridge_report.txt'}")
        return

    channels: list[ChannelData] = []
    if args.csv_alpha:
        channels.append(load_csv(args.csv_alpha))
    if args.csv_vis:
        channels.append(load_csv(args.csv_vis))
    if not channels:
        parser.error("Provide --demo or at least one --csv-alpha / --csv-vis")

    kc = extract_kc_from_channels(channels)
    bridge = build_xi_bridge(kc, a_vac_m=args.a_vac)
    lines = format_xi_bridge(bridge)

    if not args.skip_gravity:
        fc = forecast_gravity_from_xi(bridge.consensus_xi_m, r_s_hat=args.rs_hat)
        lines.extend(format_gravity_forecast(fc))

    report = "\n".join(lines) + "\n"
    print(report)
    out_path = OUTPUT / "ch_lab_xi_bridge_report.txt"
    out_path.write_text(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
