import numpy as np
from IPython.display import Audio
import librosa
import random

def random_splice(song_path, snippets):
    samples, sampling_rate = librosa.load(song_path, sr=44100, mono=True, duration=180)
    length = len(samples) #2138112
    clips = []
    for i in range(snippets):
        start = random.randint(0, length - 2)
        end = start + random.randint(start, length)
        clips.append(samples[start:end])
    return clips