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
import Utils

def load_audio(filename):
    """
    Loads a song from a file.
    """
    return librosa.load(filename, sr=44100, mono=True,)
    
def tape_audio(seconds=1):
    """
    Records a song from the microphone.
    """
    print(f"Beginnning recording for {seconds} seconds...")
    frames, sample_rate = record_audio(seconds)
    print("Finished recording.")
    samples = np.hstack([np.frombuffer(frame, dtype=np.int16) for frame in frames])
    return samples, sample_rate
def tape_and_save_audio(seconds=1, filename="test.wav"):
    """
    Records a song from the microphone and saves it to a file.
    """
    samples, sample_rate = tape_audio(seconds)
    Utils.save_wave(samples, sample_rate, filename)
    return samples, sample_rate