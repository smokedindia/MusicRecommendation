import json
import multiprocessing as mp
import os
import random
import shutil

import librosa
import numpy as np
import tqdm
import wget
from audioread import NoBackendError
from scipy.io.wavfile import write


class DatasetParams:
    def __init__(self, dataset_config):
        self.music_root = dataset_config['music_root']
        self.save_root = dataset_config['save_root']

        self.music_link = dataset_config['music_link']

        self.version = dataset_config['version']

        self.sr = dataset_config['sr']

        self.feature_size = dataset_config.get('feature_size', 1)
        self.hop_size = dataset_config.get('hop_size', self.feature_size)


def make_database(dataset_param):
    """
    If database/GTZAN/genres_original is not made,
    this function automatically generates this folder.
    """
    music_link = dataset_param.music_link
    if not os.path.isdir("./genres"):
        if not os.path.isfile("./genres.tar.gz"):
            wget.download(music_link)
        os.system("tar -xf genres.tar.gz")
    os.system("rm -r ./genres/*.mf")
    if not os.path.isdir("./database"):
        os.system("mkdir database")
    if not os.path.isdir("mkdir database/GTZAN"):
        os.system("mkdir database/GTZAN")
    if not os.path.isdir("mkdir database/GTZAN/genres_original"):
        os.system("mkdir database/GTZAN/genres_original")
    os.system("cp -r ./genres/* ./database/GTZAN/genres_original/")


class SnippetGenerator:
    """Audio snippet generator that provides iterator for audio snippet"""
    ENERGY_THRESHOLD = 1e-4

    def __init__(self, audios, sr, feature_size, hop_size):
        """

        :param audios: list of audios (not sliced)
        :param sr: sampling rate in Hz
        :param feature_size: length of audio snippet in second(s)
        :param hop_size: hop size of audio snippet in second(s)
        """
        self.audios = audios
        if hop_size == -1:
            hop_size = feature_size
        self.hop_size_samples = int(hop_size * sr)

        self.feature_size_samples = int(feature_size * sr)

        self.num_snippets = [
            int(
                (len(audio) - self.feature_size_samples)
                // self.hop_size_samples + 1
            ) for audio in audios]
        self.idx = self.__construct_idx()

    def __len__(self):
        return len(self.idx)

    def __construct_idx(self):
        """
        construct index in form (song_idx, start_sample, end_sample)
        :return: list of indices
        """
        idx = []
        for i, num_snippet in enumerate(self.num_snippets):
            idx.extend(
                [(i, k * self.hop_size_samples,
                  k * self.hop_size_samples + self.feature_size_samples)
                 for k in range(num_snippet)]
            )
        return idx

    def generate(self):
        """
        Yield audio snippets if above energy threshold
        Apply augmentation if available
        :return: audio snippet
        """
        for song_idx, start_sample, end_sample in self.idx:
            audio_snippet = self.audios[song_idx][start_sample: end_sample]

            if np.power(audio_snippet, 2).mean() > self.ENERGY_THRESHOLD:
                yield audio_snippet

    def get_random_snippet(self):
        """
        Get random audio snippet
        :return: random audio snippet
        """
        while True:
            song_idx, start_sample, end_sample = random.choice(self.idx)
            audio_snippet = self.audios[song_idx][start_sample: end_sample]

            if np.power(audio_snippet, 2).mean() > self.ENERGY_THRESHOLD:
                return audio_snippet


class DatasetGenerator:
    TEST_MODE = False
    LOAD_AUDIO_NUM_WORKERS = 16
    GEN_DATA_NUM_WORKERS = 32

    def __init__(self, dataset_config):
        self.dataset_param = DatasetParams(dataset_config)
        self.save_root = os.path.join(self.dataset_param.save_root,
                                      'version_%s' %
                                      self.dataset_param.version)

        if os.path.exists(self.save_root):
            print('path: %s exists, deleting' % self.save_root)
            shutil.rmtree(self.save_root)
        os.makedirs(self.save_root)
        if not os.path.isdir(self.dataset_param.music_root):
            make_database(self.dataset_param)
        dirs = os.listdir(self.dataset_param.music_root)
        # music_paths = {}
        # for directory in dirs:
        #     music_paths[directory] = self.__get_all_paths(
        #         os.path.join(self.dataset_param.music_root,
        #                      directory))

        # if self.TEST_MODE:
        #     music_paths = music_paths[:20]

        music_audios = {}
        for genre in dirs:
            music_paths = self.__get_all_paths(
                os.path.join(self.dataset_param.music_root,
                             genre))
            music_audios[genre] = self.__load_audios(music_paths,
                                                     audio_type=genre)

        self.music_snip_generator = {}
        for genre in music_audios.keys():
            self.music_snip_generator[genre] = SnippetGenerator(
                audios=music_audios[genre],
                sr=self.dataset_param.sr,
                feature_size=self.dataset_param.feature_size,
                hop_size=self.dataset_param.hop_size
            )

        self.count = 0

    def gen_dataset(self):
        metadata_file = 'metadata.json'
        meta = {}
        for genre in self.music_snip_generator.keys():
            meta.update(
                self.__gen_dataset(
                    mix_func=self.mix_func,
                    audio_generator=self.music_snip_generator[genre],
                    genre=genre
                )
            )

        metadata_dir = os.path.join(self.save_root, metadata_file)
        with open(metadata_dir, 'w') as f:
            json.dump(meta, f, indent=4)

    @staticmethod
    def mix_func(audio1: np.ndarray):
        return audio1

    @staticmethod
    def __get_all_paths(directory):
        return [os.path.join(directory, filename)
                for filename in os.listdir(directory)]

    def __load_audio(self, audio_path):
        try:
            return librosa.load(audio_path,
                                sr=self.dataset_param.sr)[0]
        except NoBackendError:
            print('Invalid sound recognized: %s' % audio_path)

    def __load_audios(self, audio_paths, audio_type):
        """used when multiple cpus found"""
        with mp.pool.ThreadPool(self.LOAD_AUDIO_NUM_WORKERS) as pool:
            audios = list(
                tqdm.tqdm(pool.imap_unordered(self.__load_audio, audio_paths),
                          total=len(audio_paths),
                          desc='Loading %s' % audio_type)
            )
        results = []
        for audio in audios:
            if audio is not None:
                results.append(audio)

        return results

    def __gen_dataset(self, audio_generator, mix_func, genre):
        with mp.pool.ThreadPool(self.GEN_DATA_NUM_WORKERS) as pool:
            music_ds = list(
                tqdm.tqdm(
                    pool.imap_unordered(mix_func,
                                        audio_generator.generate()),
                    total=len(audio_generator),
                    desc='Creating %s' % genre
                )
            )

        meta = {}
        for data in music_ds:
            file_id = str(self.count)
            filename = file_id + '.wav'
            write(filename=os.path.join(self.save_root, filename),
                  rate=self.dataset_param.sr,
                  data=data)
            meta[file_id] = {
                'filename': file_id,
                'label': genre
            }
            self.count += 1

        return meta
