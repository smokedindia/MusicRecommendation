import copy
import json
import multiprocessing as mp
import os
import shutil

import librosa
import numpy as np
import tqdm
from dataclasses import dataclass, asdict
from sklearn import preprocessing


@dataclass
class FeatureParams:
    def __init__(self, feature_config, **_extras):
        super(FeatureParams, self).__init__(**_extras)
        self.feature_type = feature_config["feature_type"]
        self.save_root = feature_config['save_root']
        self.sr = feature_config['sr']

        self.feature_param = None
        if self.feature_type == 'mel':
            self.n_fft = feature_config['n_fft']
            self.win_length = feature_config.get('win_length', None)
            self.hop_length = feature_config.get('hop_length', None)
            self.n_mels = feature_config['n_mels']
            self.power = feature_config.get('power', 2.0)
        elif self.feature_type == 'cqt':
            self.feature_param = CQTParams(feature_config['hop_length'],
                                           feature_config['n_bins'],
                                           feature_config['bins_per_octave'],
                                           feature_config['fmin_note'])
        elif self.feature_type == 'stft':
            self.feature_param = STFTParams(feature_config['hop_length'],
                                            feature_config['n_fft'])
        else:
            raise ValueError('Invalid feature called')


@dataclass
class CQTParams:
    hop_length: int
    n_bins: int
    bins_per_octave: int
    fmin_note: str = 'C2'
    fmin: float = librosa.note_to_hz(fmin_note)

    def to_dict(self):
        dict_attr = asdict(self)
        del dict_attr['fmin_note']
        return dict_attr

@dataclass
class STFTParams:
    hop_length: int
    n_fft: int

    def to_dict(self):
        dict_attr = asdict(self)
        return dict_attr


class FeatureExtractor:
    def __init__(self, config_list, metadata_file='metadata.json',
                 raw_meta=None):
        self.data_root = 'dataset_raw/version_%s' % \
                         config_list.dataset_config['version']
        self.feature_params = FeatureParams(config_list.feature_config)
        version = '.'.join([config_list.dataset_config['version'],
                            config_list.feature_config['version']])
        self.version_path = os.path.join(self.feature_params.save_root,
                                         'version_%s' % version)
        # options = {'mel': self.logmelspec}
        self.metadata_file = metadata_file

        if raw_meta is not None:
            self.audio_meta = raw_meta
            self.feature_meta = {}
            self.save = False
        else:
            with open(os.path.join(self.data_root, metadata_file), 'r') as f:
                self.audio_meta = json.load(f)
            self.save = True

        options = {'mel': self.melspec,
                   'cqt': self.cqt,
                   'stft': self.stft}
        self.feature_func = options[self.feature_params.feature_type]

    def extract(self):
        """performs feature extraction process"""
        if self.save:
            if not os.path.isdir(self.feature_params.save_root):
                os.mkdir(self.feature_params.save_root)

            if os.path.exists(self.version_path):
                print('feature version exists, deleting')
                shutil.rmtree(self.version_path)
            os.mkdir(self.version_path)

        audio_ids = list(self.audio_meta.keys())

        # data_list.remove('config.json')
        with mp.pool.ThreadPool(processes=8) as pool:
            metas = list(tqdm.tqdm(
                pool.imap_unordered(self.load_and_transform, audio_ids),
                total=len(audio_ids),
                desc='extracting feature')
            )

        meta = {}
        for m in metas:
            if m is not None:
                meta.update(m)
        if self.save:
            with open(os.path.join(
                    self.version_path, self.metadata_file), 'w') as f:
                json.dump(meta, f, indent=4)
        else:
            return meta

    def load_and_transform(self, audio_id):
        # .wav data containing nan values raise error. filtered by try and
        # except
        if self.save:
            audio_filename = self.audio_meta[audio_id]['filename']
            try:
                y, sr = librosa.load(os.path.join(
                    self.data_root,
                    audio_filename
                ),
                    sr=self.feature_params.sr)

            except librosa.util.exceptions.ParameterError:
                return
        else:
            audio_filename = self.audio_meta[audio_id]
            y = self.audio_meta[audio_id].pop('data')

        if y.max() == y.min():
            return

        y_feature = self.feature_func(y)

        scaler = preprocessing.StandardScaler()
        y_feature_norm = scaler.fit_transform(y_feature)

        meta = copy.deepcopy(self.audio_meta[audio_id])
        if y_feature is not None:
            if self.save:
                feature_filename = audio_filename.rstrip('.wav')
                np.save(os.path.join(self.version_path, feature_filename),
                        y_feature)
                meta['filename'] = feature_filename + '.npy'
            else:
                meta.update({'spectrogram': y_feature_norm})
            return {audio_id: meta}
        return

    def melspec(self, y):
        """transforms signal to mel spectrogram followed by log scaling
        operation """
        y_mel = librosa.feature.melspectrogram(
            y=y, sr=self.feature_params.sr,
            n_fft=self.feature_params.n_fft,
            win_length=self.feature_params.win_length,
            hop_length=self.feature_params.hop_length,
            n_mels=self.feature_params.n_mels,
            power=self.feature_params.power)

        # clip values lower than 1e-7 and log-scaling
        # if all values are below 1e-7, data is not appropriate for training
        if np.max(y_mel) < 1e-7:
            return None
        else:
            y_mel_clipped = np.clip(y_mel, 1e-7, None)
            y_mel_clipped_log = np.log10(y_mel_clipped)
        return y_mel_clipped_log

    def cqt(self, y):
        """transforms signal to constant Q transform spectrogram with log
        scaling"""
        y_cqt = np.abs(
            librosa.cqt(y=y, sr=self.feature_params.sr,
                        **self.feature_params.feature_param.to_dict())
        )
        if np.max(y_cqt) < 1e-7:
            return None
        else:
            y_cqt_clip = np.clip(y_cqt, 1e-7, None)
            y_cqt_clip_log = np.log(y_cqt_clip)
            return y_cqt_clip_log

    def stft(self, y):
        y_stft = librosa.stft(y=y,
                              **self.feature_params.feature_param.to_dict()
                              )
        if np.max(y_stft) < 1e-7:
            return None
        else:
            y_stft_clip = np.clip(y_stft, 1e-7, None)
            y_stft_clip_log = np.log(y_stft_clip)
            return y_stft_clip_log
