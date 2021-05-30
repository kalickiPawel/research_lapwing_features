import os

from kaggle.api.kaggle_api_extended import KaggleApi


class FileLoader:
    ROOT_DIR = os.path.dirname(os.path.abspath(__name__))

    def __init__(self, dataset, dest_name):
        self.dataset = dataset
        self.dest_name = dest_name
        self.dest_path = os.path.join(self.ROOT_DIR, self.dest_name)
        if not self.check_output_exist():
            if self.mk_out():
                print(f"'{os.path.basename(os.path.normpath(self.dest_path))}' dir created!")

    def is_empty(self):
        if os.path.exists(self.dest_path) and not os.path.isfile(self.dest_path):
            if not os.listdir(self.dest_path):
                print(f"{self.__class__.__name__}: Empty directory")
                return True
            else:
                print(f"{self.__class__.__name__}: Files are in directory")
                return False
        else:
            print(f"{self.__class__.__name__}: The path is either for a file or not valid")
            return False

    def check_output_exist(self):
        return os.path.exists(self.dest_path)

    def mk_out(self):
        if not os.path.exists(self.dest_path):
            os.mkdir(self.dest_path)
            return True
        return False


class DownloadKaggle(FileLoader):
    def __init__(self, dataset, dest_name):
        super().__init__(dataset, dest_name)

        if self.is_empty():
            self.load()

    def load(self):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset, path=self.dest_path, unzip=True, quiet=False)
