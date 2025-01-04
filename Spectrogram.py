import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import iterate_structure

from typing import Tuple, Callable, List
import librosa

def spectrogram(waveform: np.ndarray, plot=False):
    def fourier_complex_to_real(
    complex_coeffs: np.ndarray, N: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts complex-valued Fourier coefficients (of 
        real-valued data) to the associated amplitudes and 
        phase-shifts of the real-valued sinusoids
    
        Parameters
        ----------
        complex_coeffs : numpy.ndarray, shape-(N//2 + 1,)
        The complex valued Fourier coefficients for k=0, 1, ...
    
        N : int
        The number of samples that the DFT was performed on.
    
        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
        (amplitudes, phase-shifts)
        Two real-valued, shape-(N//2 + 1,) arrays
        """
        amplitudes = np.abs(complex_coeffs) / N

        # |a_k| = 2 |c_k| / N for all k except for
        # k=0 and k=N/2 (only if N is even)
        # where |a_k| = |c_k| / N
        amplitudes[1 : (-1 if N % 2 == 0 else None)] *= 2

        phases = np.arctan2(-complex_coeffs.imag, complex_coeffs.real)
        return amplitudes, phases
    
    

    recorded_audio = waveform
    sampling_rate = 44100
    times = np.arange(len(recorded_audio)) / sampling_rate  # <COGLINE>
    N = len(recorded_audio)
    T = N / sampling_rate
    ck = np.fft.rfft(recorded_audio)
    
    ak = np.abs(ck) / N
    ak[1 : (-1 if N % 2 == 0 else None)] *= 2
    
    freqs = np.arange(len(ak)) / T

    amps, phases = fourier_complex_to_real(ck, N)
    F = freqs.max()
    extent = (0, T, 0, F)
    if plot == True:
        fig, ax = plt.subplots()
        spectrogram, freqs, times, im = ax.specgram(
            recorded_audio,
            NFFT=4096,
            Fs=sampling_rate,
            window=mlab.window_hanning,
        )
        np.clip(spectrogram, 1E-20, a_max = None, out = spectrogram)
        S = np.log(spectrogram)
        
        plt.colorbar(im)
    else: 
        spectrogram, freqs, times = mlab.specgram(
            recorded_audio,
            NFFT=4096,
            Fs=sampling_rate,
            window=mlab.window_hanning,
        )
        np.clip(spectrogram, 1E-20, a_max = None, out = spectrogram)
        S = np.log(spectrogram)
   
    
    return spectrogram

def spectrogram_from_file(AudioFile: str) -> np.ndarray:
    recorded_audio, sampling_rate = librosa.load(AudioFile, sr=44100, mono=True)
    return spectrogram(recorded_audio, False)