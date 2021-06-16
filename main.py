from modules import DownloadKaggle, LapwingRecognition
from modules import PrepareData


if __name__ == "__main__":
    dest, bird = 'output', 'Fringilla'

    DownloadKaggle(dataset='rtatman/british-birdsong-dataset', dest_name=dest)
    data = PrepareData(dest_name=dest)
    df = data.audio_data

    lap = LapwingRecognition(frame_size=1024, n_mel=40, win_type='hann')

    results = {}

    results = lap.load_results(df.loc[df['genus'] == bird], True, results)
    results = lap.load_results(df.loc[df['genus'] != bird], False, results)

    X_train, y_train, X_test, y_test = lap.split_to_test_train(results)

    print(f"Num of samples in train set: {X_train.shape}")
    print(f"Num of samples in test set: {X_test.shape}")

    lap.run_classify(X_train, y_train, X_test, y_test)
