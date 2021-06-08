import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MusicRecommendationDataset(Dataset):
    def __init__(self, root, transform, feature_meta=None):
        self.transform = transform
        self.root = root
        self.genres = {'blues': 0, 'classical': 1, 'country': 2,
                       'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                       'pop': 7, 'reggae': 8, 'rock': 9}

        if feature_meta is not None:
            self.feature_meta = feature_meta
            self.ids = list(self.feature_meta.keys())
        else:
            with open(os.path.join(root, 'metadata.json'), 'rb') as f:
                self.feature_meta = json.load(f)
            self.ids = list(self.feature_meta.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        try:
            spectrogram = self.feature_meta[str(item)]['spectrogram']
        except KeyError:
            spectrogram = np.load(
                os.path.join(self.root,
                             self.feature_meta[str(item)]['filename'])
            )
        label = self.genres[self.feature_meta[str(item)]['label']]
        data = {'spectrogram': spectrogram,
                'label': label}

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor:
    def __call__(self, data, *args, **kwargs):
        spectrogram = data['spectrogram']
        label = data['label']
        return {'spectrogram': torch.from_numpy(spectrogram).unsqueeze(0),
                'label': label}
