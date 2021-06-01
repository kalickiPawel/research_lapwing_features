from modules import DownloadKaggle
from modules import PrepareData

if __name__ == "__main__":
    dest = 'output'

    DownloadKaggle(dataset='rtatman/british-birdsong-dataset', dest_name=dest)
    data = PrepareData(dest_name=dest)
