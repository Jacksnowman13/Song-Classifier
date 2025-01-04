from email.policy import default
import imp
import Utils
import SoundLoader
import Spectrogram
import PeakFinder
import FanOut

class Song:


    def __init__(self, title, artist, release_date, filepath):
        self.title = title
        self.artist = artist
        self.date = release_date
        self.filepath = filepath

    def __repr__(self):
        return self.title + " by " + self.artist + " released in " + str(self.date)

def get_song(ID):
    return song_info.get(ID, "Not Found in Database")

import pickle
song_info = {}
#store data
#with open('filename.pickle', 'wb') as handle:
   # pickle.dump(song_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
def add_song(title, artist, release_date):
    #load data
   # with open('filename.pickle', 'rb') as handle:
       # song_info = pickle.load(handle)
    tune = Song(title, artist, release_date, f"Songs/{title}.mp3")
    song_info[len(song_info) + 1] = tune
   # with open('filename.pickle', 'wb') as handle:
        #pickle.dump(song_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def get_song_peaks(ID):
    print(song_info[ID].filepath)
    return PeakFinder.get_peaks(Spectrogram.spectrogram(SoundLoader.load_audio(song_info[ID].filepath)[0]))
def get_fanout(ID):
    print(song_info[ID].filepath)
    return FanOut.generate_tuples(PeakFinder.get_peaks(Spectrogram.spectrogram(SoundLoader.load_audio(song_info[ID].filepath)[0])))
    
