# zdf_moog_ladder_tanh
```python
"""
zdf_moog_ladder_tanh.py

Zero-delay-feedback Moog ladder (4-pole lowpass) using:
- TPT / trapezoidal 1-pole blocks
- Ladder instantaneous response (G ξ + S) for zero-delay feedback
- Tanh OTA-style nonlinearity in the feedback summing node
- Functional style: no classes, no dicts; only arrays + tuples
- tick() = one-sample step
- process() = lax.scan over a block
- __main__ has simple smoke tests + plots

Math follows Zavalishin's TPT 1-pole and ladder filter derivation:
- 1-pole instantaneous response: y = G x + S
- 4-pole cascade instantaneous response: y = G4 u + S_ladder
- Nonlinear zero-delay feedback solved by Newton iterations per sample.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import matplotlib.pyplot as plt


Array = jax.Array


# ============================================================
# TPT 1-pole lowpass core (trapezoidal integrator)
#
# Continuous-time model: dy/dt = ωc (x - y)
# Discretized via TPT / bilinear transform.
#
# Parameter g0 is the BLT / TPT integrator coefficient:
#   g0 = tan(pi * fc / fs)
#
# Implementation is the canonical TPT 1-pole step:
#   G = g0 / (1 + g0)
#   v = (x - s) * G
#   y = v + s
#   s' = y + v
#
# State 's' is the integrator state (z^-1 output).
# ============================================================

@jax.jit
def tpt_onepole_tick(x: Array, s: Array, g0: Array) -> Tuple[Array, Array]:
    """
    One-sample TPT 1-pole lowpass tick.

    Parameters
    ----------
    x : Array
        Input sample.
    s : Array
        Integrator state (z^-1 output of trapezoidal integrator).
    g0 : Array
        TPT integrator coefficient, typically g0 = tan(pi * fc / fs).

    Returns
    -------
    s_new : Array
        Updated state.
    y     : Array
        Lowpass output sample.
    """
    G = g0 / (1.0 + g0)          # instantaneous slope for 1-pole
    v = (x - s) * G
    y = v + s
    s_new = y + v
    return s_new, y


# ============================================================
# ZDF Moog Ladder with tanh OTA nonlinearity
#
# Structure:
#   x --(+)-> v = x - k*y
#         |
#       nonlin(v) = tanh(drive * v) = u
#         |
#         v
#      4 x TPT 1-pole lowpass
#         |
#         y  (ladder output, 24 dB/oct lowpass)
#
# BUT: since the feedback is zero-delay, y depends on u (same sample),
# and u depends on y via v = x - k*y. We solve the scalar nonlinear
# equation per-sample using Newton iterations, leveraging the linear
# instantaneous response of the 4-pole chain:
#
#   y = G4 * u + S_ladder
#   u = tanh(drive * (x - k * y))
#
# with G4 and S_ladder constructed from the 4 identical TPT 1-poles.
#
# State layout (tuple of arrays):
#   (fs, g0, k, drive, v_prev, s1, s2, s3, s4)
#
# All scalars are 0-dim jax.Arrays; s1..s4 are integrator states.
# ============================================================


def moog_ladder_init(
    fs: float,
    cutoff_hz: float,
    resonance: float = 0.0,
    drive: float = 1.0,
) -> Tuple[Array, ...]:
    """
    Initialize state for ZDF Moog ladder with tanh OTA nonlinearity.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    cutoff_hz : float
        Cutoff frequency in Hz (0 < cutoff_hz < fs/2).
    resonance : float, optional
        Resonance feedback coefficient k. For the linear ladder,
        infinite resonance is at k ≈ 4. Here, with nonlinearity,
        values in [0, 4] are typical.
    drive : float, optional
        Input differential pair drive for tanh. Higher values =
        stronger saturation in the loop.

    Returns
    -------
    state : tuple of Arrays
        (fs, g0, k, drive, v_prev, s1, s2, s3, s4)
    """
    fs_j = jnp.asarray(fs, jnp.float32)
    cutoff_j = jnp.asarray(cutoff_hz, jnp.float32)
    k_j = jnp.asarray(resonance, jnp.float32)
    drive_j = jnp.asarray(drive, jnp.float32)

    # TPT integrator coefficient (prewarped 1-pole)
    # g0 = tan(pi * fc / fs)
    g0 = jnp.tan(jnp.pi * cutoff_j / fs_j)

    zero = jnp.asarray(0.0, jnp.float32)
    state = (fs_j, g0, k_j, drive_j, zero, zero, zero, zero, zero)
    return state


@jax.jit
def moog_ladder_tanh_tick(
    state: Tuple[Array, ...],
    x: Array,
) -> Tuple[Tuple[Array, ...], Array]:
    """
    One-sample tick for the ZDF Moog ladder with tanh OTA nonlinearity.

    Parameters
    ----------
    state : tuple
        (fs, g0, k, drive, v_prev, s1, s2, s3, s4)
    x : Array
        Input sample (mono).

    Returns
    -------
    new_state : tuple
        Updated state with same layout.
    y : Array
        Output sample (ladder lowpass).
    """
    fs, g0, k, drive, v_prev, s1, s2, s3, s4 = state

    # Stage instantaneous slope for 1-pole: G1 = g0 / (1 + g0)
    inv_1pg = 1.0 / (1.0 + g0)
    G1 = g0 * inv_1pg

    # Instantaneous response of 4-pole cascade: y = G4 * u + S_ladder
    G4 = G1 ** 4

    # For each stage, instantaneous intercept S_i = s_i / (1 + g0)
    S1 = s1 * inv_1pg
    S2 = s2 * inv_1pg
    S3 = s3 * inv_1pg
    S4 = s4 * inv_1pg

    # Ladder intercept S_ladder = G1^3 S1 + G1^2 S2 + G1 S3 + S4
    S_ladder = (G1 ** 3) * S1 + (G1 ** 2) * S2 + G1 * S3 + S4

    # OTA-style nonlinearity in the feedback summing node
    def nonlin(v):
        return jnp.tanh(drive * v)

    def nonlin_prime(v):
        t = jnp.tanh(drive * v)
        return drive * (1.0 - t * t)

    # Solve scalar nonlinear zero-delay feedback:
    #   v = x - k * y, with y = G4 * nonlin(v) + S_ladder
    # using a couple of Newton iterations starting from v_prev.
    def newton_step(v):
        u = nonlin(v)
        y_inst = G4 * u + S_ladder
        F = v - x + k * y_inst               # F(v) = v - x + k*y
        Fp = 1.0 + k * G4 * nonlin_prime(v)  # dF/dv = 1 + k*G4*nonlin'(v)
        v_next = v - F / Fp
        return v_next

    v = v_prev
    v = newton_step(v)
    v = newton_step(v)  # 2 iterations are usually enough for audio use

    u = nonlin(v)       # final nonlinear input to ladder

    # Now run the 4 identical TPT 1-pole stages in series with input u
    s1_new, y1 = tpt_onepole_tick(u,  s1, g0)
    s2_new, y2 = tpt_onepole_tick(y1, s2, g0)
    s3_new, y3 = tpt_onepole_tick(y2, s3, g0)
    s4_new, y4 = tpt_onepole_tick(y3, s4, g0)  # ladder output

    new_state = (fs, g0, k, drive, v, s1_new, s2_new, s3_new, s4_new)
    return new_state, y4


@jax.jit
def moog_ladder_tanh_process(
    state: Tuple[Array, ...],
    x_block: Array,
) -> Tuple[Tuple[Array, ...], Array]:
    """
    Process a block of samples through the ZDF Moog ladder.

    Parameters
    ----------
    state : tuple
        Initial state (fs, g0, k, drive, v_prev, s1, s2, s3, s4).
    x_block : Array
        Input block, shape (n,).

    Returns
    -------
    final_state : tuple
        Updated state after processing the block.
    y_block : Array
        Output block, shape (n,).
    """
    def step(carry, x_t):
        return moog_ladder_tanh_tick(carry, x_t)

    final_state, y_block = lax.scan(step, state, x_block)
    return final_state, y_block


# ============================================================
# Helpers: plots (non-JAX)
# ============================================================

def plot_wave(x: Array, fs: float, title: str) -> None:
    x_np = np.asarray(x)
    n = x_np.shape[0]
    t = np.arange(n) / fs

    plt.figure(figsize=(9, 3))
    plt.plot(t, x_np, lw=1.0)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrum(x: Array, fs: float, title: str) -> None:
    x_np = np.asarray(x)
    n = x_np.shape[0]
    win = np.hanning(n)
    xw = x_np * win

    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(np.fft.rfft(xw))
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    plt.figure(figsize=(9, 3))
    plt.plot(f, mag_db, lw=1.0)
    plt.xscale("log")
    plt.xlim([20, fs / 2])
    plt.ylim([-140, 10])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Smoke tests
# ============================================================

if __name__ == "__main__":
    print("Running ZDF Moog ladder (tanh) smoke tests...")

    fs = 48_000.0
    dur = 1.0
    nframes = int(fs * dur)

    # --------------------------------------------------------
    # 1) Impulse response at moderate cutoff & resonance
    # --------------------------------------------------------
    cutoff = 1000.0   # Hz
    resonance = 0.0   # no feedback
    drive = 1.0

    state = moog_ladder_init(fs=fs, cutoff_hz=cutoff, resonance=resonance, drive=drive)

    impulse = jnp.zeros((nframes,), jnp.float32).at[0].set(1.0)
    state_imp, y_imp = moog_ladder_tanh_process(state, impulse)

    print(" ✓ Impulse test ran")

    plot_wave(y_imp, fs, f"ZDF Moog ladder (tanh): impulse, fc={cutoff} Hz, k={resonance}")
    plot_spectrum(y_imp, fs, f"ZDF Moog ladder (tanh): impulse spectrum, fc={cutoff} Hz, k={resonance}")

    # --------------------------------------------------------
    # 2) Resonant noise test to see behavior near self-oscillation
    # --------------------------------------------------------
    resonance_hi = 3.5   # close to 4, but below; nonlinearity will tame it
    drive_hi = 1.5

    state2 = moog_ladder_init(fs=fs, cutoff_hz=500.0, resonance=resonance_hi, drive=drive_hi)

    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (nframes,), dtype=jnp.float32) * 0.1

    state2_final, y_noise = moog_ladder_tanh_process(state2, noise)

    print(" ✓ Resonant noise test ran")

    plot_wave(y_noise, fs, f"ZDF Moog ladder (tanh): noise, fc=500 Hz, k={resonance_hi}, drive={drive_hi}")
    plot_spectrum(y_noise, fs, f"ZDF Moog ladder (tanh): noise spectrum, fc=500 Hz, k={resonance_hi}")
    print("Done.")

    import sounddevice as sd
    sd.play(y_noise)
    sd.wait()
```
