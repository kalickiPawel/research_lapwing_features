import sklearn

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
        x, sr = r.get('data'), r.get('samplerate')
        X = stft(x)
        Xdb = amplitude_to_db(abs(X))
        figure = plt.figure(figsize=(8, 5))
        plt.title(f"Spectogram bird id: {i}")
        spec = librosa.display.specshow(Xdb, sr=sr, x_axis='s', y_axis='hz')
        plt.colorbar(spec)
        plt.show()

        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.tight_layout(rect=[0.02, 0.03, 1, 0.95], h_pad=3.0)
        fig.suptitle(f"Bird {i}: {r.get('species')}, {r.get('english_cname')} - {r.get('country')} ==> {r.get('type')}")

        n0, n1 = 9000, 9100
        axes[0, 0].plot(x[n0:n1])
        axes[0, 0].set_title(f"Samples {n0} to {n1}")
        axes[0, 0].grid()

        axes[1, 0].plot(1, 1)

        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)

        def normalize(x, axis=0):
            return sklearn.preprocessing.minmax_scale(x, axis=axis)

        # Plotting the Spectral Centroid along the waveform
        librosa.display.waveplot(x, sr=sr, alpha=0.4, ax=axes[1, 0])
        axes[1, 0].plot(t, normalize(spectral_centroids), color='r')
        axes[1, 0].set_title("Spectral centroid")

        spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
        librosa.display.waveplot(x, sr=sr, alpha=0.4, ax=axes[2, 0])
        axes[2, 0].plot(t, normalize(spectral_rolloff), color='r')
        axes[2, 0].set_title("Spectral Rolloff")
        axes[2, 0].grid()

        mfccs = librosa.feature.mfcc(x, sr=sr)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[0, 1])
        axes[0, 1].set_title("MFCC")

        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title("MFCC Feature Scaling")

        print(mfccs.mean(axis=1))
        print(mfccs.var(axis=1))

        hop_length = 512
        chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
        librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length,
                                 cmap='coolwarm', ax=axes[2, 1])
        axes[2, 1].set_title("Chroma Frequencies")
        plt.show()

        print(f"Sum of zero crossings for bird: {i} is {sum(librosa.zero_crossings(x[n0:n1], pad=False))}")
