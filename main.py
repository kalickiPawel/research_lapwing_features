from modules import DownloadKaggle
from modules import PrepareData
from matplotlib import pyplot as plt

if __name__ == "__main__":
    dest, bird = 'output', 'Fringilla'
    
    DownloadKaggle(dataset='rtatman/british-birdsong-dataset', dest_name=dest)
    data = PrepareData(dest_name=dest)
    
    df = data.audio_data
    
    for i, r in df.loc[df['genus'] == bird].iterrows():
        Pxx, freqs, bins, im = plt.specgram(
            r.get('data'),
            Fs=r.get('samplerate')
        )
        plt.title(f"Spectogram bird {i}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

