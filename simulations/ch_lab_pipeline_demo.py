#!/usr/bin/env python3
"""
Full CH lab pipeline: protocol verdict → k_c → ξ → gravity v3 + α_G calibration.

  python ch_lab_pipeline_demo.py
  python ch_lab_pipeline_demo.py --a-vac 150e-9 --rs-hat 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ch_xi_lab_bridge import (
    build_xi_bridge,
    extract_kc_from_protocol,
    forecast_gravity_from_xi,
    format_gravity_forecast,
    format_kc_extraction,
    format_xi_bridge,
)
from control_channel_analysis import (
    OUTPUT,
    evaluate_protocol,
    plot_protocol,
    write_lab_demo_csvs,
)
from gradient_threshold_analysis import load_csv

OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CH lab → ξ → gravity end-to-end demo")
    parser.add_argument("--a-vac", type=float, default=150e-9, help="Casimir lattice period [m]")
    parser.add_argument("--rs-hat", type=float, default=1.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    alpha_csv, vis_csv, control_csv = write_lab_demo_csvs()
    alpha_ch = load_csv(alpha_csv)
    vis_ch = load_csv(vis_csv)
    control_ch = load_csv(control_csv)

    protocol = evaluate_protocol([alpha_ch, vis_ch], control_ch)
    kc = extract_kc_from_protocol(protocol)
    bridge = build_xi_bridge(kc, a_vac_m=args.a_vac)
    forecast = forecast_gravity_from_xi(bridge.consensus_xi_m, r_s_hat=args.rs_hat)

    print("=" * 72)
    print("CH LAB PIPELINE — protocol → ξ → gravity")
    print("=" * 72)
    print(f"\nVerdict: {protocol.verdict.value}\n")
    for reason in protocol.reasons:
        print(f"  • {reason}")

    print("\n" + "\n".join(format_kc_extraction(kc)))
    print("\n" + "\n".join(format_xi_bridge(bridge)))
    print("\n".join(format_gravity_forecast(forecast)))

    report_path = OUTPUT / "ch_lab_pipeline_report.txt"
    lines = [
        "CH lab pipeline report",
        f"Verdict: {protocol.verdict.value}",
        "",
        *protocol.reasons,
        "",
        *format_kc_extraction(kc),
        "",
        *format_xi_bridge(bridge),
        *format_gravity_forecast(forecast),
    ]
    report_path.write_text("\n".join(lines) + "\n")

    if not args.no_plot:
        plot_protocol(protocol, OUTPUT / "ch_lab_pipeline_protocol.png")
        print(f"\nWrote {OUTPUT / 'ch_lab_pipeline_protocol.png'}")

    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
