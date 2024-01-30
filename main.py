import argparse
from audioset_download import Downloader


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-path", type=str, default="", required=True, help="")
    parser.add_argument("--n-jobs", type=int, default=12, help="")
    parser.add_argument("--download-type", type=str, default="balanced_train", help="")
    parser.add_argument(
        "--copy-and-replicate", default=False, action="store_true", help=""
    )
    parser.add_argument("--csv-path", type=str, default=None, help="")
    parser.add_argument("--format", type=str, default="wav", help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    d = Downloader(
        root_path=args.root_path,
        labels=None,
        n_jobs=args.n_jobs,
        download_type=args.download_type,
        copy_and_replicate=args.copy_and_replicate,
        csv_path=args.csv_path,
    )
    d.download(format=args.format)
