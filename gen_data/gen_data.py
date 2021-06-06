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
        self.noise_root = dataset_config['noise_root']
        self.save_root = dataset_config['save_root']

        self.music_link = dataset_config['music_link']
        self.noise_link = dataset_config['noise_link']

        self.version = dataset_config['version']

        self.sr = dataset_config['sr']

        self.feature_size = dataset_config.get('feature_size', 1)
        self.hop_size = dataset_config.get('hop_size', self.feature_size)

        self.snr_low = dataset_config.get('snr_low', None)
        self.snr_high = dataset_config.get('snr_high', None)
        self.snr_hop_size = dataset_config.get('snr_hop_size', 1)


def make_database(dataset_param):
    """
    If database is not made (music and noise),
    this function automatically generates this folder.
    """
    music_link = dataset_param.music_link
    dataset_path = os.path.join("./", dataset_param.music_root)
    music_zip_file = "./genres.tar.gz"
    music_unzip_folder = "./genres"
    if os.path.isdir(dataset_path) and \
            len(os.listdir(dataset_path)) == 0:
        shutil.rmtree(dataset_path)
    if not os.path.isdir(dataset_path):
        if os.path.isdir(music_unzip_folder) and \
                len(os.listdir(music_unzip_folder)) == 0:
            shutil.rmtree(music_unzip_folder)
        if not os.path.isdir(music_unzip_folder):
            if not os.path.isfile(music_zip_file):
                print("Start downloading dataset...")
                wget.download(music_link)
                print("Finished downloading")
            print("Start unzipping genre folder..")
            os.system("tar -xf genres.tar.gz")
            print("Finished unzipping genre folder")

        print("Start copying data files...")
        shutil.copytree(music_unzip_folder, dataset_path)
        print("Finished copying data files")

    noise_link = dataset_param.noise_link
    noise_path = os.path.join("./", dataset_param.noise_root)
    noise_zip_file = "./ESC-50-master.zip"
    noise_unzip_folder = "./ESC-50-master/audio"

    if os.path.isdir(noise_path) and \
            len(os.listdir(noise_path)) == 0:
        shutil.rmtree(dataset_path)

    if not os.path.isdir(noise_path):
        if os.path.isdir(noise_unzip_folder) and \
                len(os.listdir(noise_unzip_folder)) == 0:
            shutil.rmtree(noise_unzip_folder)
        if not os.path.isdir(noise_unzip_folder):
            if not os.path.isfile(noise_zip_file):
                print("Start downloading noise...")
                wget.download(noise_link)
                print("Finished downloading")
            print("Start unzipping noise folder...")
            os.system("unzip -oq ESC-50-master")
            print("Finished unzipping")
        print("Start copying noise data...")
        shutil.copytree(noise_unzip_folder, noise_path)
        print("Finished copying")


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
            if len(self.idx) == 0:
                audio_snippet = self.audios[
                    random.randint(0, len(self.audios) - 1)
                ]
            else:
                song_idx, start_sample, end_sample = random.choice(self.idx)
                audio_snippet = self.audios[song_idx][start_sample: end_sample]

            if np.power(audio_snippet, 2).mean() > self.ENERGY_THRESHOLD:
                return audio_snippet


class DatasetGenerator:
    TEST_MODE = False
    LOAD_AUDIO_NUM_WORKERS = 16
    GEN_DATA_NUM_WORKERS = 32

    def __init__(self, dataset_config, save):
        self.dataset_param = DatasetParams(dataset_config)
        if save:
            self.save_root = os.path.join(self.dataset_param.save_root,
                                          'version_%s' %
                                          self.dataset_param.version)

            if os.path.exists(self.save_root):
                print('path: %s exists, deleting' % self.save_root)
                shutil.rmtree(self.save_root)
            os.makedirs(self.save_root)

        make_database(self.dataset_param)
        dirs = os.listdir(self.dataset_param.music_root)
        dirs = [valid_dir for valid_dir in dirs if valid_dir.find('mf') == -1]

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

        noise_paths = self.__get_all_paths(self.dataset_param.noise_root)
        noise_audios = self.__load_audios(noise_paths, audio_type="noise")
        self.noise_snip_generator = SnippetGenerator(
            audios=noise_audios,
            sr=self.dataset_param.sr,
            feature_size=self.dataset_param.feature_size,
            hop_size=-1
        )

        self.count = 0
        self.save = save

    def gen_dataset(self):
        metadata_file = 'metadata.json'
        meta = {}
        for genre in self.music_snip_generator.keys():
            meta.update(
                self.__gen_dataset(
                    mix_func=self.mix_audio
                    if self.dataset_param.snr_low is not None else self.no_mix,
                    audio_generator=self.music_snip_generator[genre],
                    genre=genre
                )
            )

        if self.save:
            metadata_dir = os.path.join(self.save_root, metadata_file)
            with open(metadata_dir, 'w') as f:
                json.dump(meta, f, indent=4)
        else:
            return meta

    @staticmethod
    def no_mix(audio1: np.ndarray):
        """
        Simply returns the audio if mixing SNR is not defined
        :param audio1: audio data 1
        :return:
            same audio data 1
        """
        return audio1, None

    def mix_audio(self, audio1: np.ndarray):
        """
        Mix audio and noise with random SNR in range
        :param audio1: audio data 1
        :return:
            mixed audio data, SNR
        """
        noise = self.noise_snip_generator.get_random_snippet()

        snr = random.randrange(self.dataset_param.snr_low,
                               self.dataset_param.snr_high + 1, self.dataset_param.snr_hop_size)

        rms1 = np.sqrt(np.sum(np.square(audio1) / len(audio1)))
        rms2 = np.sqrt(np.sum(np.square(noise) / len(noise)))

        audio1_norm = rms2 / rms1 * audio1

        audio1_gain = 10 ** (snr / 20) * audio1_norm

        if len(audio1_gain) > len(noise):
            mixed_audio = audio1_gain
            mixed_audio[:len(noise)] += noise
        else:
            mixed_audio = audio1_gain + noise

        return mixed_audio, snr

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
                    pool.imap_unordered(mix_func, audio_generator.generate()),
                    total=len(audio_generator),
                    desc='Creating %s' % genre
                )
            )

        meta = {}
        for audio_snr in music_ds:
            audio, snr = audio_snr
            meta_key = str(self.count)
            filename = meta_key + '.wav'
            meta[meta_key] = {
                'filename': filename,
                'label': genre,
                'snr': snr
            }
            if self.save:
                write(filename=os.path.join(self.save_root, filename),
                      rate=self.dataset_param.sr,
                      data=np.array(audio))
            else:
                meta[meta_key].update({'data': audio})

            self.count += 1

        return meta
