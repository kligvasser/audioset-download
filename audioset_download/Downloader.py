import os
import joblib
import pandas as pd


class Downloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """

    def __init__(
        self,
        root_path: str,
        labels: list = None,  # None to download all the dataset
        n_jobs: int = 1,
        download_type: str = "unbalanced_train",
        copy_and_replicate: bool = True,
    ):
        """
        This method initializes the class.
        :param root_path: root path of the dataset
        :param labels: list of labels to download
        :param n_jobs: number of parallel jobs
        :param download_type: type of download (unbalanced_train, balanced_train, eval)
        :param copy_and_replicate: if True, the audio file is copied and replicated for each label.
                                    If False, the audio file is stored only once in the folder corresponding to the first label.
        """
        # Set the parameters
        self.root_path = root_path
        self.labels = labels
        self.n_jobs = n_jobs
        self.download_type = download_type
        self.copy_and_replicate = copy_and_replicate

        # Create the path
        os.makedirs(self.root_path, exist_ok=True)
        self.read_class_mapping()

    def read_class_mapping(self):
        """
        This method reads the class mapping.
        :return: class mapping
        """

        class_df = pd.read_csv(
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
            sep=",",
        )

        self.display_to_machine_mapping = dict(
            zip(class_df["display_name"], class_df["mid"])
        )
        self.machine_to_display_mapping = dict(
            zip(class_df["mid"], class_df["display_name"])
        )
        return

    def download(self, format="vorbis", quality=5, csv_path=None):
        """
        This method downloads the dataset using the provided parameters.
        :param format: format of the audio file (vorbis, mp3, m4a, wav), default is vorbis
        :param quality: quality of the audio file (0: best, 10: worst), default is 5
        """

        self.format = format
        self.quality = quality

        # Load the metadata
        if csv_path:
            metadata = pd.read_csv(
                csv_path,
                sep=", ",
                skiprows=3,
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
        else:
            metadata = pd.read_csv(
                f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.download_type}_segments.csv",
                sep=", ",
                skiprows=3,
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
        if self.labels is not None:
            self.real_labels = [
                self.display_to_machine_mapping[label] for label in self.labels
            ]
            metadata = metadata[
                metadata["positive_labels"].apply(
                    lambda x: any([label in x for label in self.real_labels])
                )
            ]
            # remove " in the labels
        metadata["positive_labels"] = metadata["positive_labels"].apply(
            lambda x: x.replace('"', "")
        )
        metadata = metadata.reset_index(drop=True)

        print(f"Downloading {len(metadata)} files...")

        # Download the dataset
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(
                metadata.loc[i, "YTID"],
                metadata.loc[i, "start_seconds"],
                metadata.loc[i, "end_seconds"],
                metadata.loc[i, "positive_labels"],
            )
            for i in range(len(metadata))
        )

        print("Done.")

    def download_file(
        self,
        ytid: str,
        start_seconds: float,
        end_seconds: float,
        positive_labels: str,
    ):
        """
        This method downloads a single file. It only download the audio file at 16kHz.
        If a file is associated to multiple labels, it will be stored multiple times.
        :param ytid: YouTube ID.
        :param start_seconds: start time of the audio clip.
        :param end_seconds: end time of the audio clip.
        :param positive_labels: labels associated with the audio clip.
        """

        # Create the path for each label that is associated with the file
        if self.copy_and_replicate:
            for label in positive_labels.split(","):
                display_label = self.machine_to_display_mapping[label]
                os.makedirs(os.path.join(self.root_path, display_label), exist_ok=True)
        else:
            display_label = self.machine_to_display_mapping[
                positive_labels.split(",")[0]
            ]
            os.makedirs(os.path.join(self.root_path, display_label), exist_ok=True)

            # Download the file using yt-dlp
            # store in the folder
            os.system(
                f'yt-dlp -x --audio-format {self.format} --audio-quality {self.quality} --output "{os.path.join(self.root_path, ytid)}.%(ext)s" --postprocessor-args "-ss {start_seconds} -to {end_seconds}" https://www.youtube.com/watch?v={ytid}'
            )
        return
