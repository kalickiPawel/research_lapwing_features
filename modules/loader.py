import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi


class FileLoader:
    ROOT_DIR = os.path.dirname(os.path.abspath(__name__))

    def __init__(self, dest_name):
        self.data_name = dest_name
        self.data_path = os.path.join(self.ROOT_DIR, self.data_name)
        if not self.check_output_exist():
            if self.mk_out():
                print(f"'{os.path.basename(os.path.normpath(self.data_path))}' dir created!")

    def is_empty(self):
        if os.path.exists(self.data_path) and not os.path.isfile(self.data_path):
            if not os.listdir(self.data_path):
                print(f"{self.__class__.__name__}: \tEmpty directory")
                return True
            else:
                print(f"{self.__class__.__name__}: \tFiles are in directory")
                return False
        else:
            print(f"{self.__class__.__name__}: \tThe path is either for a file or not valid")
            return False

    def check_output_exist(self):
        return os.path.exists(self.data_path)

    def mk_out(self):
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            return True
        return False


class DownloadKaggle(FileLoader):
    """
    Service to download files from Kaggle
    Require Kaggle API token in hidden file in home directory.
    dataset -> name of dataset in Kaggle,
    dest_name -> name of directory where will be placed files from dataset
    """

    def __init__(self, dest_name, dataset):
        super().__init__(dest_name)
        start = time.time()
        self.dataset = dataset
        if self.is_empty():
            self.load()
        print(f"{self.__class__.__name__}: \t{(time.time() - start):.5f}s \t\tDownloaded dataset from Kaggle")

    def load(self):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset, path=self.data_path, unzip=True, quiet=False)
