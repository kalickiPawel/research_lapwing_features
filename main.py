from scipy.signal import find_peaks
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from modules import DownloadKaggle
from modules import PrepareData
from matplotlib import pyplot as plt
from librosa import stft, amplitude_to_db
import librosa.display
import numpy as np
import seaborn as sns


if __name__ == "__main__":
    dest, bird = 'output', 'Fringilla'

    DownloadKaggle(dataset='rtatman/british-birdsong-dataset', dest_name=dest)
    data = PrepareData(dest_name=dest)

    df = data.audio_data

    frame_size = 1024
    hop_length = int(0.75 * frame_size)
    results = {}

    for i, r in df.loc[df['genus'] == bird].iterrows():
        x, sr = r.get('data'), r.get('samplerate')
        X = stft(x)
        Xdb = amplitude_to_db(abs(X))
        figure = plt.figure(figsize=(8, 5))
        plt.title(f"Spectogram bird id: {i}")
        spec = librosa.display.specshow(Xdb, sr=sr, x_axis='s', y_axis='hz')
        plt.colorbar(spec)
        # plt.show()

        signal_out = np.array([max(x[i:i + frame_size]) for i in range(0, len(x), hop_length)])
        t = librosa.frames_to_time(range(0, signal_out.size), hop_length=hop_length)

        peaks, _ = find_peaks(signal_out, prominence=0.4)
        if len(peaks) >= 10:
            results[f"lapwing_{i}"] = t[peaks]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_title(f"Envelope bird id: {i}")
        librosa.display.waveplot(x, ax=ax, alpha=0.5)
        ax.set_ylim((-1, 1))
        ax.plot(t, signal_out, color='r')
        ax.plot(t[peaks], signal_out[peaks], "x")
        # plt.show()

    for i, r in df.loc[df['genus'] != bird].iterrows():
        x, sr = r.get('data'), r.get('samplerate')
        signal_out = np.array([max(x[i:i + frame_size]) for i in range(0, len(x), hop_length)])
        t = librosa.frames_to_time(range(0, signal_out.size), hop_length=hop_length)
        peaks, _ = find_peaks(signal_out, prominence=0.4)
        if len(peaks) >= 10:
            results[f"not_lapwing_{i}"] = t[peaks]

    X_train, y_train = [], []
    X_test, y_test = [], []
    i, j = 0, 0

    for k, v in results.items():
        if 'not_lapwing' in k:
            if i < 3:
                X_train.append(v[:10])
                y_train.append(1)
            else:
                X_test.append(v[:10])
                y_test.append(1)
            i += 1
        elif 'lapwing' in k:
            if j % 2 == 0:
                if j != 0:
                    X_train.append(v[:10])
                    y_train.append(0)
            else:
                X_test.append(v[:10])
                y_test.append(0)
            j += 1

    X_train, y_train = np.array(X_train, dtype=object), np.array(y_train, dtype='bool')
    X_test, y_test = np.array(X_test, dtype=object), np.array(y_test, dtype='bool')

    print(f"Num of samples in train set: {X_train.shape}")
    print(f"Num of samples in test set: {X_test.shape}")

    classifier = SVC(kernel='rbf', random_state=1)
    classifier.fit(X_train, y_train)

    counter = 0
    for i in range(1000):
        y_pred = classifier.predict(X_test)
        counter += y_pred[0] is False
    print(counter)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    accuracy = float(cm.diagonal().sum()) / len(y_test)
    print("\nAccuracy Of SVM For The Given Dataset: ", accuracy)
