import os
import re
import time
import pandas as pd
import soundfile as sf

from modules import FileLoader


class PrepareData(FileLoader):
    def __init__(self, dest_name):
        super().__init__(dest_name)
        start = time.time()
        self.meta_path = os.path.join(self.data_path, [_ for _ in os.listdir(self.data_path) if _.endswith(".csv")][0])
        self.meta_data = self.load_meta()
        self.audio_data = self.load_audio_files()
        print(f"{self.__class__.__name__}: \t\t{(time.time() - start):.5f}s \t\tData prepared to usage")

    def load_meta(self):
        return pd.read_csv(self.meta_path, index_col="file_id")

    def load_audio_files(self):
        regex = re.compile(r'\d+')

        d = {int(x): sf.read(os.path.join(root, f)) for root, _, fs in os.walk(
            self.data_path
        ) for f in fs for x in regex.findall(f) if f.endswith(".flac")}

        return self.meta_data.join(pd.DataFrame.from_dict(d, orient='index', columns=['data', 'samplerate']))
