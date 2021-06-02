from modules import DownloadKaggle
from modules import PrepareData
from matplotlib import pyplot as plt
from librosa import stft, amplitude_to_db
import librosa.display

if __name__ == "__main__":
    dest, bird = 'output', 'Fringilla'
    
    DownloadKaggle(dataset='rtatman/british-birdsong-dataset', dest_name=dest)
    data = PrepareData(dest_name=dest)
    
    df = data.audio_data
    
    for i, r in df.loc[df['genus'] == bird].iterrows():
        X = stft(r.get('data'))
        Xdb = amplitude_to_db(abs(X))
        plt.figure()
        plt.title(f"Spectogram bird {i}")
        librosa.display.specshow(Xdb, sr=r.get('samplerate'), x_axis='s', y_axis='hz')
        plt.colorbar()
        plt.show()
