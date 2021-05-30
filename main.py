from modules import DownloadKaggle

if __name__ == "__main__":
    # Service to download files from Kaggle
    # Require Kaggle API token in hidden file in home directory.
    # dataset -> name of dataset in Kaggle,
    # dest_name -> name of directory where will be placed files from dataset
    DownloadKaggle(dataset='rtatman/british-birdsong-dataset', dest_name='output')
