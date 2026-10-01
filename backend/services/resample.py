"""Band-limited PCM16 resampling for the telephony bridge.

Piper's voices are all 22050Hz; Plivo (per the calling platform's own codec
docstring) wants 24000Hz out. Naive linear interpolation was tried and rejected
in that same calling platform's codebase for exactly this kind of resample --
its own comment: "aliases badly and adds a metallic edge that listeners
[notice]". This implements a proper windowed-sinc (band-limited) interpolator
instead: for each output sample, a small, fixed-width kernel of neighboring
input samples weighted by a Hann-windowed sinc, which is what a polyphase
resampler computes too, just evaluated directly per output sample rather than
via precomputed per-phase filter tables -- no scipy dependency (not already a
project dependency; see requirements.txt), and simple enough to verify by
listening rather than trusting the math alone.
"""

import numpy as np

# Half-width of the sinc kernel in INPUT samples -- 8 gives 16 taps total, a
# reasonable quality/speed tradeoff for resampling already-synthesized speech
# over a phone line (itself band-limited), not a high-fidelity source.
_KERNEL_HALF_WIDTH = 8


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resamples 16-bit mono PCM bytes from from_rate to to_rate."""
    if from_rate == to_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return pcm

    ratio = from_rate / to_rate
    out_length = max(1, int(round(samples.size / ratio)))
    # Position, in INPUT sample units, that each output sample falls at.
    positions = np.arange(out_length, dtype=np.float64) * ratio
    base = np.floor(positions).astype(np.int64)

    taps = np.arange(-_KERNEL_HALF_WIDTH + 1, _KERNEL_HALF_WIDTH + 1)  # length 2*HALF_WIDTH
    # (out_length, taps) matrices -- each row is the kernel sample offsets/
    # distances for one output sample.
    indices = base[:, None] + taps[None, :]
    distances = positions[:, None] - indices

    # Windowed sinc: sinc(d) shaped by a Hann window over the kernel's width,
    # so the filter rolls off smoothly instead of truncating sharply (a sharp
    # truncation is exactly what produces the "metallic edge" artifact).
    with np.errstate(invalid="ignore"):
        sinc = np.sinc(distances)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * (taps - taps[0] + 0.5) / len(taps))
    weights = sinc * window[None, :]
    weights /= weights.sum(axis=1, keepdims=True)

    clipped_indices = np.clip(indices, 0, samples.size - 1)
    gathered = samples[clipped_indices]
    resampled = np.sum(gathered * weights, axis=1)

    return np.clip(resampled, -32768, 32767).astype("<i2").tobytes()
