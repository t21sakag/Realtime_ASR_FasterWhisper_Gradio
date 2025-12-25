from typing import Tuple

import numpy as np
import scipy.signal


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    abs_max = np.abs(audio).max()
    audio = audio.astype(np.float32)
    if abs_max > 0:
        audio *= 1 / 32768
    return audio.squeeze()


def resample_to_16k(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    if sample_rate == 16000:
        return audio, sample_rate
    num_samples = int(len(audio) * 16000 / sample_rate)
    resampled = scipy.signal.resample(audio, num_samples)
    return resampled, 16000
