import matplotlib.pyplot as plt
import numpy as np
import librosa
from microphone import record_audio
import matplotlib.mlab as mlab
from IPython.display import Audio
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import iterate_structure
from typing import Tuple, Callable, List
import SoundLoader
import SongDatabase
import PeakFinder
import Spectrogram
import Splice
import FanOut

#Boueny's Audio Library
def save_audio(audio, filename="test"):
    """
    Save an audio signal to a file.
    """
    with open(filename, "wb") as f:
        f.write(audio.data)
def save_wave(samples, sample_rate=44100, filename="test.wav"):
    save_audio(Audio(samples, rate=sample_rate), filename)
def display_sound(times, sound):
    fig, ax = plt.subplots()
    ax.plot(times, sound)
    ax.set_ylabel("Pressure [Pa]")
    ax.set_xlabel("Time [s]")
    return fig, ax
def display_waveform(times, waveform, frequency: float, periods=-1, sampling_rate=44100):
    t=times
    pressures=waveform
    if(periods != -1):
        seconds = periods * (1 / frequency)
        t = t[:int(seconds * sampling_rate)]
        pressures = pressures[:int(seconds * sampling_rate)]
    return display_sound(t, pressures)
def pressure(times: np.ndarray, *, amp: float, freq: float, phase_shift) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * times + phase_shift)
def fourier_complex_to_real(
    complex_coeffs: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray]:
    amplitudes = np.abs(complex_coeffs) / N
    amplitudes[1 : (-1 if N % 2 == 0 else None)] *= 2

    phases = np.arctan2(-complex_coeffs.imag, complex_coeffs.real)
    return amplitudes, phases
def file_to_peaks(filepath):
    return PeakFinder.get_peaks(Spectrogram.spectrogram(SoundLoader.load_audio(filepath)[0]))
def file_to_fan(filepath):
    return FanOut.generate_tuples(PeakFinder.get_peaks(Spectrogram.spectrogram(SoundLoader.load_audio(filepath)[0])))
def wave_to_fan(waveform):
    return FanOut.generate_tuples(PeakFinder.get_peaks(Spectrogram.spectrogram(waveform)))