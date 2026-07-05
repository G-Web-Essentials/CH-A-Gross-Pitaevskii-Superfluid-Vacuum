#!/usr/bin/env python3
"""
CH real-units analysis — uses measured physical constants and published limits.

No synthetic "planted" signals. Outputs actual SI numbers and pass/fail vs experiments.

  python ch_real_units_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path

# --- CODATA-style constants (SI) ---
C = 299_792_458.0                    # m/s
HBAR = 1.054_571_817e-34             # J·s
G_MEAS = 6.674_30e-11                # m³ kg⁻¹ s⁻²
RHO_LAMBDA = 9.9e-27                 # kg/m³ (~Ω_Λ ≈ 0.69 × ρ_crit, ρ_crit~9.5e-27)

# Published experimental order-of-magnitude bounds
DC_OVER_C_SIDEREAL = 1e-15           # typical modern Michelson/resonator (conservative)
BETA_FERMI_MAX = 5e14                # quadratic LV scale ~ E_QG ~ 1e11 GeV (order-of-mag.)
EOTWASH_ALPHA_MAX_AT_1MM = 0.003     # |α| upper bound ~mm range (Lee 2020 ballpark)
CASIMIR_RIPPLE_ALPHA_MAX = 0.01      # no established ripple >~1% at 50–600 nm (conservative)


def ch_G_from_rho_xi(rho_in: float, xi: float, c_s: float = C, alpha_g: float = 1.0) -> float:
    """G = alpha_G * c_s² * xi / rho_in  [SI]."""
    return alpha_g * c_s**2 * xi / rho_in


def ch_rho_in_from_G_xi(xi: float, c_s: float = C, alpha_g: float = 1.0) -> float:
    """Invert Eq. (6): rho_in = alpha_G * c_s² * xi / G."""
    return alpha_g * c_s**2 * xi / G_MEAS


def grain_mass(xi: float, c_s: float = C) -> float:
    """m_grain = hbar * c_s / (c² * xi)  [kg]."""
    return HBAR * c_s / (C**2 * xi)


def omega_0(xi: float, c_s: float = C) -> float:
    return c_s / xi


def epsilon_for_lambda(rho_in: float, rho_lambda: float = RHO_LAMBDA) -> float:
    """ε = ρ_Λ / ρ_in — fraction of inertial density that may gravitate as Λ."""
    return rho_lambda / rho_in


def healing_length_gpe(rho_in: float, m_grain: float, g_coupling: float) -> float:
    """ξ = hbar / sqrt(2 m g rho) — check internal GPE consistency."""
    return HBAR / math.sqrt(2 * m_grain * g_coupling * rho_in)


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("CHRONOS-HYDRODYNAMICS — REAL UNITS ANALYSIS")
    print("Uses measured G, c, ℏ, ρ_Λ and published experimental bounds")
    print("=" * 70)

    # --- Scan healing length ξ (real meters) ---
    xi_values = [
        ("Planck scale", 1.616e-35),
        ("1 nm", 1e-9),
        ("150 nm (Casimir demo scale)", 150e-9),
        ("1 mm", 1e-3),
        ("Schwarzschild Moon ~0.11 mm", 0.11e-3),
        ("1 m", 1.0),
    ]

    print("\n--- 1. CH parameters from measured G (Eq. 6: G = α_G c² ξ / ρ_in) ---\n")
    print(f"{'ξ (label)':<32} {'ξ [m]':<12} {'ρ_in [kg/m³]':<14} {'ε for ρ_Λ':<12} {'m_grain [kg]':<12}")
    print("-" * 82)

    rows = []
    for label, xi in xi_values:
        rho_in = ch_rho_in_from_G_xi(xi)
        eps = epsilon_for_lambda(rho_in)
        m_g = grain_mass(xi)
        rows.append((label, xi, rho_in, eps, m_g))
        print(f"{label:<32} {xi:<12.3e} {rho_in:<14.3e} {eps:<12.3e} {m_g:<12.3e}")

    print("\nInterpretation:")
    print("  • ρ_in is INERTIAL density fixed by (G, c, ξ).")
    print("  • ε = ρ_Λ/ρ_in is how small the gravitating fraction must be to match dark energy.")
    print("  • Large ξ → small ρ_in → larger ε (easier cosmological constant).")
    print("  • Small ξ → huge ρ_in → ε ~ 10⁻⁵⁰ or smaller (two-component split essential).")

    # --- Consistency check at ξ = 1 mm ---
    xi = 1e-3
    rho_in = ch_rho_in_from_G_xi(xi)
    m_g = grain_mass(xi)
    f0 = omega_0(xi)
    g_calc = ch_G_from_rho_xi(rho_in, xi)
    print("\n--- 2. Dimensional self-check at ξ = 1 mm ---\n")
    print(f"  ρ_in     = {rho_in:.4e} kg/m³")
    print(f"  G_calc   = {g_calc:.4e} m³/(kg·s²)  (G_meas = {G_MEAS:.4e})")
    print(f"  m_grain  = {m_g:.4e} kg")
    print(f"  ω₀       = {f0:.4e} rad/s  ({f0/(2*math.pi):.4e} Hz)")
    print(f"  Match G:  {abs(g_calc - G_MEAS)/G_MEAS * 100:.2f}% (by construction)")

    # --- vs experiments ---
    print("\n--- 3. CH scenarios vs REAL experimental bounds ---\n")

    tests = []

    # Sidereal anisotropy: uniform CH predicts null
    tests.append((
        "Sidereal δc/c (#1)",
        "CH uniform vacuum → predict null",
        f"|δc/c| < ~{DC_OVER_C_SIDEREAL:.0e} (observed)",
        "PASS for CH (null expected)",
    ))

    # GRB dispersion
    tests.append((
        "GRB β (#4)",
        f"Strong dispersion β > {BETA_FERMI_MAX:.0e}",
        f"Fermi-like bound β ≲ {BETA_FERMI_MAX:.0e}",
        "FAIL if CH predicts large β",
    ))

    # Fifth force at 1 mm
    tests.append((
        "Yukawa at λ=1 mm (#5)",
        f"|α| > {EOTWASH_ALPHA_MAX_AT_1MM}",
        f"Eöt-Wash |α| ≲ {EOTWASH_ALPHA_MAX_AT_1MM}",
        "FAIL for gravity-strength fifth force",
    ))

    # Casimir ripple
    tests.append((
        "Casimir ripple (#3)",
        f"α_ripple > {CASIMIR_RIPPLE_ALPHA_MAX} always-on",
        "No established >1% ripple 50–600 nm",
        "FAIL for strong always-on ripple; gradient-gated may evade",
    ))

    print(f"{'Test':<22} {'CH strong claim':<28} {'Real bound':<32} {'Verdict':<30}")
    print("-" * 112)
    for name, ch, bound, verdict in tests:
        print(f"{name:<22} {ch:<28} {bound:<32} {verdict:<30}")

    # --- Gradient threshold: what |∇ρ| means in SI ---
    print("\n--- 4. Gradient scale (Prediction #7) — order-of-magnitude SI ---\n")
    print("  |∇ρ| has units kg/m⁴.")
    print("  Examples of macroscopic density gradients (NOT vacuum ρ_in directly):")
    # Newtonian tidal gradient of air density is wrong - use gravitational potential gradient
    # For illustration: Δρ/Δx across Casimir gap
    d_casimir = 100e-9  # 100 nm gap
    # If CH ripple turns on when vacuum responds - we don't have measured vacuum ∇ρ
    # Use Earth's vertical air density scale height ~8 km: dρ_air/dz ~ ρ/H
    rho_air = 1.2
    H = 8000.0
    grad_air = rho_air / H
    print(f"  Earth atmosphere:     |∇ρ_air| ~ ρ/H ~ {grad_air:.3e} kg/m⁴")
    print(f"  Casimir gap Δd=100nm:  lab knob changes boundary conditions (no direct ∇ρ_in meas.)")
    print("  → #7 needs a defined proxy for |∇ρ| (gap, tidal ∇Φ, mass proximity).")
    print("  → Laptop can compute thresholds once CH maps proxy → |∇ρ| in SI.")

    # --- What laptop CANNOT do ---
    print("\n--- 5. What requires hardware (not laptop-only) ---\n")
    for item in [
        "Measure Casimir force (pN at nm gaps)",
        "Generate entangled photons + Bell statistics",
        "Matter-wave interferometry visibility",
        "Sub-mm fifth-force torsion balance",
        "Direct vacuum ρ_in or ξ measurement",
    ]:
        print(f"  ✗ {item}")

    print("\n--- 6. What laptop CAN do with real units ---\n")
    for item in [
        "This script: CH (ρ_in, ε, m_grain) from measured G and chosen ξ",
        "ch_vs_published_limits.py: plot CH curves vs Eöt-Wash / Fermi",
        "Analyze public Fermi GRB FITS for dispersion (needs download)",
        "LLR ephemeris residuals (public NASA/APOLLO archives)",
    ]:
        print(f"  ✓ {item}")

    # Save summary table
    summary = out_dir / "ch_real_units_summary.txt"
    with summary.open("w") as f:
        f.write("CH real-units analysis\n")
        f.write(f"G_meas = {G_MEAS}\n")
        f.write(f"rho_Lambda = {RHO_LAMBDA} kg/m^3\n\n")
        for label, xi, rho_in, eps, m_g in rows:
            f.write(f"{label}: xi={xi:.3e} m, rho_in={rho_in:.3e}, epsilon={eps:.3e}, m_grain={m_g:.3e} kg\n")
    print(f"\nSaved {summary}")
    print("=" * 70)


if __name__ == "__main__":
    main()
