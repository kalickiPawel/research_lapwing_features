import librosa
import librosa.display

import pandas
import numpy as np
import seaborn as sns
import noisereduce as nr

from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from tqdm import tqdm, trange
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def get_spec(x, fs):
    f, t, Sxx = signal.spectrogram(x, fs)
    return f, t, Sxx


def draw_peaks(s, s_out, t, peaks, ax):
    ax.set_ylim((-1, 1))
    librosa.display.waveplot(s, ax=ax, alpha=0.5)
    ax.set_ylim((-1, 1))
    ax.plot(t, s_out, color='r')
    ax.plot(t[peaks], s_out[peaks], "x")


def draw_spec(x, fs, ax):
    f, t, Sxx = get_spec(x, fs)
    ax.pcolormesh(t, f, Sxx, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')


class LapwingRecognition:
    def __init__(self, frame_size: int, n_mel: int, win_type: str):
        self.frame_size = frame_size
        self.hop_length = int(0.75 * frame_size)
        self.n_mel = n_mel
        self.win_type = win_type

    def load_results(self, df: pandas.DataFrame, type_of_set: bool, results: dict):
        for i, r in tqdm(df.iterrows(), total=df.shape[0]):
            x, fs = r.get('data'), r.get('samplerate')
            x = nr.reduce_noise(audio_clip=x, noise_clip=x, prop_decrease=1, verbose=False)
            sig_out, t, peaks, = self.find_onsets(x, prominance=0.4)

            if type_of_set:
                fig, ax = plt.subplots(3, 1, figsize=(8, 5))
                plt.suptitle(f"{r.get('genus')} no.{i}")
                librosa.display.waveplot(x, ax=ax[0], alpha=0.5)
                draw_peaks(x, sig_out, t, peaks, ax[1])
                draw_spec(x, fs, ax[2])
                plt.show()

            # TODO: Use peaks and spec to create characteristic feature (then change t[peaks] to something other)

            f, t, Sxx = get_spec(x, fs)
            if len(peaks) > 0:
                if type_of_set:
                    results[f"lapwing_{i}"] = t[peaks]
                else:
                    results[f"not_lapwing_{i}"] = t[peaks]

        return results

    def find_onsets(self, x, prominance):
        s_out = np.array([max(x[i:i + self.frame_size]) for i in range(0, len(x), self.hop_length)])
        t = librosa.frames_to_time(range(0, s_out.size), hop_length=self.hop_length)
        peaks, _ = find_peaks(s_out, prominence=prominance)
        return s_out, t, peaks

    def split_to_test_train(self, results: dict):
        print(len([f for f in results.keys() if 'lapwing' in f]))
        # cout_lapwing = len('lapwing' in results.keys())
        # count_not_lapwing =
        x_train, y_train = [], []
        X_test, y_test = [], []
        position_of_lapwing = 0
        i, j = 0, 0

        for k, v in results.items():
            if 'not_lapwing' in k:
                if i < 3:
                    x_train.append(v if isinstance(v, float) else v[:10])
                    y_train.append(0)
                else:
                    X_test.append(v if isinstance(v, float) else v[:10])
                    y_test.append(0)
                i += 1
            elif 'lapwing' in k:
                if j % 2 == 0:
                    if j != 0:
                        x_train.append(v if isinstance(v, float) else v[:10])
                        y_train.append(1)
                else:
                    X_test.append(v if isinstance(v, float) else v[:10])
                    y_test.append(1)
                    position_of_lapwing = j
                j += 1

        x_train, y_train = np.array(x_train, dtype=object), np.array(y_train, dtype='bool')
        x_test, y_test = np.array(X_test, dtype=object), np.array(y_test, dtype='bool')
        return x_train, y_train, X_test, y_test, position_of_lapwing

    def run_classify(self, x_train, y_train, x_test, y_test):
        pass
        # classifier = SVC(kernel='rbf', random_state=1)
        # classifier.fit(x_train, y_train)
        #
        # counter = 0
        # for i in trange(1000):
        #     y_pred = classifier.predict(x_test)
        #     counter += y_pred[0] is False
        # print(counter)
        #
        # y_pred = classifier.predict(x_test)
        # cm = confusion_matrix(y_test, y_pred)
        # sns.heatmap(cm)
        # plt.xlabel('true label')
        # plt.ylabel('predicted label')
        # plt.show()
        #
        # accuracy = float(cm.diagonal().sum()) / len(y_test)
        # print("\nAccuracy Of SVM For The Given Dataset: ", accuracy)
