"""
Prediction #2: Orientation-dependent Bell (CHSH) correlations with sidereal modulation.

Simulates S = E(a,b) - E(a,b') + E(a',b) + E(a',b') for entangled qubits,
with optional supersolid-vacuum modulation: S_eff(t) = S0 + dS*cos(omega_s*t + phi0).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Sidereal angular frequency (rad/s): 2*pi / (23 h 56 min)
OMEGA_S = 2 * np.pi / (23 * 3600 + 56 * 60)
S_QUANTUM = 2 * np.sqrt(2)


def correlation(a: float, b: float, visibility: float = 1.0) -> float:
    """Quantum correlation for singlet: E = -V*cos(2(a-b))."""
    return -visibility * np.cos(2 * (a - b))


def measure_chsh(
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
    visibility: float = 1.0,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> float:
    rng = rng or np.random.default_rng()
    s = (
        correlation(a, b, visibility)
        - correlation(a, b_prime, visibility)
        + correlation(a_prime, b, visibility)
        + correlation(a_prime, b_prime, visibility)
    )
    return s + rng.normal(0, noise)


def run_sidereal_sim(
    n_bins: int = 48,
    trials_per_bin: int = 500,
    delta_s: float = 0.05,
    phi0: float = 0.7,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sidereal hours, mean S per bin, standard error per bin."""
    rng = np.random.default_rng(seed)
    a, a_prime = 0.0, np.pi / 4
    b, b_prime = np.pi / 8, 3 * np.pi / 8

    hours = np.linspace(0, 24, n_bins, endpoint=False)
    t_sec = hours * 3600

    means = np.zeros(n_bins)
    errs = np.zeros(n_bins)

    for i, t in enumerate(t_sec):
        s0 = S_QUANTUM + delta_s * np.cos(OMEGA_S * t + phi0)
        samples = [
            measure_chsh(a, a_prime, b, b_prime, visibility=0.95, noise=0.03, rng=rng)
            for _ in range(trials_per_bin)
        ]
        samples = np.array(samples)
        # Scale samples toward modulated set-point (phenomenological demo)
        samples = samples - np.mean(samples) + s0
        means[i] = np.mean(samples)
        errs[i] = np.std(samples, ddof=1) / np.sqrt(trials_per_bin)

    return hours, means, errs


def run_null_sim(n_bins: int = 48, trials_per_bin: int = 500, seed: int = 7):
    return run_sidereal_sim(n_bins, trials_per_bin, delta_s=0.0, seed=seed)


def main() -> None:
    hours_sig, means_sig, errs_sig = run_sidereal_sim(delta_s=0.06)
    hours_null, means_null, errs_null = run_null_sim()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].errorbar(hours_sig, means_sig, yerr=errs_sig, fmt="o-", capsize=3, label="Supersolid modulation")
    axes[0].axhline(S_QUANTUM, color="gray", ls="--", label=r"$2\sqrt{2}$ (max quantum)")
    axes[0].axhline(2.0, color="red", ls=":", label="Classical bound")
    axes[0].set_xlabel("Sidereal hour")
    axes[0].set_ylabel("CHSH parameter S")
    axes[0].set_title("Prediction #2: Bell S vs sidereal time (simulated signal)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(hours_null, means_null, yerr=errs_null, fmt="s-", capsize=3, color="green", label="Standard QFT (null)")
    axes[1].axhline(S_QUANTUM, color="gray", ls="--")
    axes[1].set_xlabel("Sidereal hour")
    axes[1].set_ylabel("CHSH parameter S")
    axes[1].set_title("Null hypothesis: flat S within noise")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT / "bell_sidereal_sim.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Fourier peak at sidereal frequency
    residual = means_sig - np.mean(means_sig)
    fft = np.fft.rfft(residual)
    freqs = np.fft.rfftfreq(len(residual), d=24 / len(residual))  # cycles per hour
    idx = np.argmax(np.abs(fft[1:])) + 1
    print(f"Dominant Fourier component: {freqs[idx]:.4f} cycles/hour (expect ~1/24 ≈ 0.0417)")


if __name__ == "__main__":
    main()
