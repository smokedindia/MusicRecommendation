import json
import multiprocessing as mp
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset


class MusicRecommendationDataset(Dataset):
    def __init__(self, root, transform, load_mem):
        self.transform = transform
        self.root = root
        self.load_mem = load_mem
        with open(os.path.join(self.root, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.ids = list(self.metadata.keys())

        if self.load_mem:
            with mp.pool.ThreadPool(processes=16) as pool:
                self.data_list = list(
                    tqdm.tqdm(pool.imap_unordered(self.load, self.ids),
                              total=len(self.ids),
                              desc='Loading data')
                )
        self.genres = {'blues': 0, 'classical': 1, 'country': 2,
                       'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                       'pop': 7, 'reggae': 8, 'rock': 9}

    def load(self, id_element):
        spectrogram = np.load(os.path.join(self.root,
                                           self.metadata[id_element]['filename']))
        label = np.zeros(10)
        label[self.genres[self.metadata[id_element]['label']]] = 1
        data = {'spectrogram': spectrogram,
                'label': label}
        return data

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if self.load_mem:
            data = self.data_list[item]
        else:
            data = self.load(id_element=self.ids[item])

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor:
    def __call__(self, data, *args, **kwargs):
        spectrogram = data['spectrogram']
        label = data['label']
        return {'spectrogram': torch.from_numpy(spectrogram).unsqueeze(0),
                'label': label.astype(np.longlong)}
