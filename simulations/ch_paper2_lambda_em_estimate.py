"""
Paper 2 — grain-scale estimate of λ_em, κ, and α_max vs detectability.

Refined polarizability (Tier 1.2c): see ch_paper2_grain_polarizability.py.
Primary model: κ_classical = η* e² c²/(ε₀ ℏ c_s³) — ξ-independent.

  python ch_paper2_lambda_em_estimate.py
  python ch_paper2_lambda_em_estimate.py --k-heal 2.0
  python ch_paper2_lambda_em_estimate.py --xi-sweep
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ch_dispersion_core import CHParams
from ch_paper2_grain_polarizability import (
    KAPPA_BENCHMARK,
    PolarizabilityModel,
    alpha_max_from_kappa_linear,
    build_polarizability_models,
    preferred_kappa_band,
    summarize_xi_independence,
)
from ch_paper2_kappa_derive import (
    ALPHA_MAX_PAPER1,
    kappa_for_target_alpha,
    lambda_em_from_kappa,
    predict_alpha_max,
)
from ch_paper2_supersolid_ground_state import (
    minimize_ground_state_staged,
    representative_d_hat,
)

OUTPUT = Path(__file__).parent / "output" / "paper2"

POWER_FLOORS = {
    "optimistic (σ=0.005)": 0.0188,
    "moderate (σ=0.010)": 0.0373,
    "conservative (σ=0.020)": 0.0741,
}

DEFAULT_XI_SWEEP = [10e-9, 20e-9, 50e-9, 100e-9]


@dataclass
class GrainCouplingEstimate:
    name: str
    alpha_grain_m3: float
    kappa: float
    lambda_em: float
    alpha_max_pred: float
    notes: str


def models_to_estimates(
    models: list[PolarizabilityModel],
    alpha_at_012: float,
) -> list[GrainCouplingEstimate]:
    out: list[GrainCouplingEstimate] = []
    for m in models:
        if not m.use_in_preferred and "cross-check" in m.notes:
            continue
        out.append(
            GrainCouplingEstimate(
                name=m.name,
                alpha_grain_m3=m.alpha_grain_m3,
                kappa=m.kappa,
                lambda_em=lambda_em_from_kappa(m.kappa),
                alpha_max_pred=alpha_max_from_kappa_linear(m.kappa, alpha_at_012),
                notes=f"{m.xi_scaling}; {m.notes}",
            )
        )
    return out


def add_preferred_band_rows(
    rows: list[dict],
    band,
    alpha_at_012: float,
) -> None:
    for label, kappa in [
        ("Preferred κ low (0.5× classical)", band.kappa_low),
        ("Preferred κ mid (classical)", band.kappa_mid),
        ("Preferred κ high (CM cap)", band.kappa_high),
    ]:
        alpha = alpha_max_from_kappa_linear(kappa, alpha_at_012)
        rows.append(
            {
                "source": label,
                "alpha_max": alpha,
                "alpha_pct": 100.0 * alpha,
                "kappa": kappa,
                "vs_moderate_floor": alpha / POWER_FLOORS["moderate (σ=0.010)"],
                "notes": "Tier 1.2c preferred band",
            }
        )


def build_detectability_rows(
    ch: CHParams,
    gs,
    estimates: list[GrainCouplingEstimate],
    alpha_at_012: float,
    band,
) -> list[dict]:
    rows: list[dict] = []

    def add_row(label: str, alpha: float, kappa: float | None, source: str) -> None:
        rows.append(
            {
                "source": label,
                "alpha_max": alpha,
                "alpha_pct": 100.0 * alpha,
                "kappa": kappa if kappa is not None else "",
                "vs_moderate_floor": alpha / POWER_FLOORS["moderate (σ=0.010)"],
                "notes": source,
            }
        )

    add_row("Paper 2 derived (κ=0.12 benchmark)", alpha_at_012, KAPPA_BENCHMARK, "mode sum + joint fit")
    add_preferred_band_rows(rows, band, alpha_at_012)

    for est in estimates:
        add_row(f"Grain model: {est.name}", est.alpha_max_pred, est.kappa, est.notes)

    invert = kappa_for_target_alpha(ch, gs, ALPHA_MAX_PAPER1)

    for label, floor in POWER_FLOORS.items():
        k_req = KAPPA_BENCHMARK * (floor / alpha_at_012) if alpha_at_012 > 0 else float("nan")
        add_row(f"90% power floor — {label}", floor, k_req, "ch_threshold_power_study.py")

    add_row("Paper 1 illustrative demo", ALPHA_MAX_PAPER1, invert["kappa_required"], "not derived; demo only")
    return rows


def write_table_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "alpha_max", "alpha_pct", "kappa", "vs_moderate_floor", "notes"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {path}")


def plot_detectability_band(
    rows: list[dict],
    alpha_at_012: float,
    band,
    out_png: Path,
    xi_nm: float,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    floor_opt = POWER_FLOORS["optimistic (σ=0.005)"]
    floor_mod = POWER_FLOORS["moderate (σ=0.010)"]
    floor_con = POWER_FLOORS["conservative (σ=0.020)"]

    ax.axhspan(floor_opt, floor_con, color="C2", alpha=0.12, label="90% power band")
    ax.axhline(ALPHA_MAX_PAPER1, color="C1", ls="--", lw=1.2, label="Paper 1 demo 12%")
    ax.axhline(alpha_at_012, color="C0", ls="-", lw=2, label=rf"$\kappa$=0.12 ({100*alpha_at_012:.1f}%)")
    ax.axhspan(
        alpha_max_from_kappa_linear(band.kappa_low, alpha_at_012),
        alpha_max_from_kappa_linear(band.kappa_high, alpha_at_012),
        color="C3",
        alpha=0.2,
        label="preferred κ band",
    )

    grain_rows = [r for r in rows if r["source"].startswith("Grain model")]
    for i, r in enumerate(grain_rows):
        ax.plot(i, r["alpha_max"], "s", color="C4", ms=8, zorder=5)

    ax.set_ylabel(r"$\alpha_{\max}$")
    ax.set_xlabel("Grain models (points)")
    ax.set_ylim(0, max(ALPHA_MAX_PAPER1 * 1.1, floor_con * 1.15))
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(rf"Detectability vs grain $\kappa$ ($\xi$={xi_nm:.0f} nm)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {out_png}")


def write_report(
    path: Path,
    ch: CHParams,
    gs,
    models: list[PolarizabilityModel],
    band,
    rows: list[dict],
    alpha_at_012: float,
) -> None:
    lines = [
        "Paper 2 — grain lambda_em estimate (Tier 1.2c refined)",
        f"xi = {ch.xi*1e9:.1f} nm",
        f"m_grain = {ch.m_grain:.6e} kg",
        f"omega_0 = {ch.omega_0:.6e} rad/s",
        f"E_xi = {ch.e_xi_gev:.6f} GeV",
        f"eta* = {gs.eta_star:.4f}, a*/(2pi xi) = {gs.ratio_to_hyp:.4f}",
        "",
        "Preferred kappa band (classical microphysics):",
        f"  kappa_low  = {band.kappa_low:.6e}",
        f"  kappa_mid  = {band.kappa_mid:.6e}  (xi-independent analytic)",
        f"  kappa_high = {band.kappa_high:.6e}  (CM / benchmark cap)",
        f"  kappa_CM   = {band.kappa_cm:.6e}",
        "",
        "Polarizability models:",
    ]
    for m in models:
        lines.extend(
            [
                f"  [{m.name}]",
                f"    alpha_grain = {m.alpha_grain_m3:.6e} m^3",
                f"    kappa = {m.kappa:.6e}",
                f"    scaling: {m.xi_scaling}",
                f"    {m.notes}",
                "",
            ]
        )
    lines.append("Detectability comparison:")
    for r in rows:
        kappa_s = f"{r['kappa']:.4e}" if r["kappa"] != "" else "—"
        lines.append(
            f"  {r['source'][:52]:52s}  alpha={r['alpha_max']:.4f}  kappa={kappa_s}"
        )
    lines.extend(
        [
            "",
            "Tier 1.2c conclusions:",
            f"- Preferred κ ≈ {band.kappa_mid:.3f} (± factor 2) from classical grain oscillator",
            f"- Maps to α_max ≈ {100*alpha_max_from_kappa_linear(band.kappa_mid, alpha_at_012):.2f}% at this xi",
            "- κ=0.12 is upper perturbative benchmark, not grain-derived",
            "- α_fs model excluded from preferred band at xi < 40 nm",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def run_xi_sweep_report(k_heal: float, n_max: int) -> None:
    """Write kappa_classical constancy check across xi grid."""
    rows = []
    for xi in DEFAULT_XI_SWEEP:
        ch = CHParams(xi=xi)
        d_hat, _ = representative_d_hat(ch)
        gs = minimize_ground_state_staged(d_hat, ch.xi, k_heal=k_heal)
        band = preferred_kappa_band(ch, gs.eta_star)
        pred = predict_alpha_max(ch, gs, KAPPA_BENCHMARK, n_max=n_max)
        alpha_mid = alpha_max_from_kappa_linear(band.kappa_mid, pred["alpha_max_pred"])
        rows.append(
            {
                "xi_nm": xi * 1e9,
                "eta_star": gs.eta_star,
                "kappa_mid": band.kappa_mid,
                "kappa_low": band.kappa_low,
                "kappa_high": band.kappa_high,
                "alpha_pref_mid_pct": 100 * alpha_mid,
                "alpha_bench_pct": 100 * pred["alpha_max_pred"],
            }
        )

    path = OUTPUT / "kappa_preferred_xi_sweep.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")

    ch_list = [CHParams(xi=x) for x in DEFAULT_XI_SWEEP]
    summary = summarize_xi_independence(ch_list, eta=0.21)
    (OUTPUT / "kappa_classical_xi_scaling.txt").write_text(summary + "\n")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grain lambda_em OOM + detectability table")
    parser.add_argument("--xi", type=float, default=50e-9)
    parser.add_argument("--k-heal", type=float, default=2.0)
    parser.add_argument("--n-max", type=int, default=800)
    parser.add_argument("--xi-sweep", action="store_true", help="Also write preferred kappa xi sweep CSV")
    args = parser.parse_args()

    ch = CHParams(xi=args.xi)
    d_hat, _ = representative_d_hat(ch)
    gs = minimize_ground_state_staged(d_hat, ch.xi, k_heal=args.k_heal)

    pred_ref = predict_alpha_max(ch, gs, kappa=KAPPA_BENCHMARK, n_max=args.n_max)
    alpha_at_012 = pred_ref["alpha_max_pred"]

    models = build_polarizability_models(ch, gs.eta_star)
    band = preferred_kappa_band(ch, gs.eta_star)
    estimates = models_to_estimates(models, alpha_at_012)
    rows = build_detectability_rows(ch, gs, estimates, alpha_at_012, band)

    tag = f"xi{int(args.xi * 1e9)}nm"
    write_table_csv(OUTPUT / f"detectability_table_{tag}.csv", rows)
    plot_detectability_band(rows, alpha_at_012, band, OUTPUT / f"detectability_band_{tag}.png", ch.xi * 1e9)
    write_report(OUTPUT / f"lambda_em_estimate_{tag}.txt", ch, gs, models, band, rows, alpha_at_012)

    if args.xi_sweep:
        run_xi_sweep_report(args.k_heal, args.n_max)

    print(f"\nDerived (κ=0.12): α_max = {100*alpha_at_012:.2f}%")
    print(f"Preferred κ band: {band.kappa_low:.4f} – {band.kappa_high:.4f} (mid={band.kappa_mid:.4f})")
    print(f"Preferred α_max ≈ {100*alpha_max_from_kappa_linear(band.kappa_mid, alpha_at_012):.2f}%")


if __name__ == "__main__":
    main()
