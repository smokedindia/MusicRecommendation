import os

import audioread.exceptions
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from main import Configs
from model import MusicRecommendationModel
from preprocessing import FeatureExtractor

GENRES = {'blues': 0, 'classical': 1, 'country': 2,
          'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
          'pop': 7, 'reggae': 8, 'rock': 9}


def audio_to_tensors(sample_path: str,
                     config_list: Configs,
                     hop_size: float) -> torch.FloatTensor:
    """

    Args:
        sample_path:
            path of test data
        config_list:
            dataclass for dataset, feature, train configuration
        hop_size:
            reference to slicing time

    Returns:
        Tensor data for model input

    """
    # initialize required parameters from config file
    extractor = FeatureExtractor(config_list=config_list, test=True)
    time_res = int(extractor.feature_params.sr /
                   extractor.feature_params.hop_length)
    feature_size = int(config_list.dataset_config['feature_size'] * time_res)
    hop_size = int(hop_size * time_res)
    pad_size = feature_size // 2

    try:
        y, sr = librosa.load(sample_path, sr=extractor.feature_params.sr)
    except audioread.exceptions.NoBackendError:
        return
    y_feature = extractor.feature_func(y)

    # scaler = StandardScaler()
    # y_feature = scaler.fit_transform(y_feature)

    y_feature_padded = np.pad(y_feature, (
        (0, 0), (feature_size - pad_size, feature_size)),
                              'constant', constant_values=0)
    n_input = int(y_feature.shape[1] // hop_size) + 1

    result = np.empty((n_input, y_feature.shape[0], feature_size))
    for i in range(n_input):
        hop_length = i * hop_size
        spec_input = y_feature_padded[:, hop_length: hop_length + feature_size]
        result[i, :, :] = spec_input

    return torch.from_numpy(result).type(torch.FloatTensor) \
        .to('cuda').unsqueeze(1)


def divide_into_batches(model_input: torch.Tensor,
                        batch_size: int = 64) -> list:
    """
    returns tensors with specified batch size to prevent memory overload
    :param model_input: tensors for model input
    :param batch_size: batch size for model input
    :return:
        batches of tensors for model input
    """
    batches = []
    num_batches = model_input.shape[0] // batch_size
    if model_input.shape[0] % batch_size:
        num_batches += 1
    for i in range(num_batches):
        batch = model_input[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)
    return batches


def run_test(config_list: Configs,
             hop_size: float) -> int:
    """

    :param config_list: configuration info dataclass
    :param hop_size: hop size of slicing test data
    :return:
        1 if successful with overall accuracy printed
    """

    # loads all the audio data from test database
    audio_list = []
    for genre in GENRES.keys():
        test_path = os.path.join('test_music', genre)
        test_files = os.listdir(test_path)
        tensor = None
        for file in test_files:
            sub_tensor = audio_to_tensors(os.path.join(test_path, file),
                                          config_list=config_list,
                                          hop_size=hop_size)
            if sub_tensor is not None:
                if tensor is None:
                    tensor = sub_tensor
                else:
                    tensor = torch.cat((tensor, sub_tensor))
        audio_list.append((tensor, genre))

    # loads model
    model_versions = ['16.1.5']
    for version in model_versions:
        state_dict = torch.load(f'models/{version}/model.ckpt')
        dims = audio_list[0][0].shape
        model = torch.nn.DataParallel(
            MusicRecommendationModel(dims[-2], dims[-1]).to('cuda')
        )
        model.load_state_dict(state_dict=state_dict)

        confusion_matrix = np.zeros(shape=(10, 10))
        for (data, label) in audio_list:
            preds = model(data)
            preds = torch.max(torch.round(preds), 1)[1]
            print(preds, GENRES[label])
            for i in range(10):
                for point in preds:
                    if point == i:
                        confusion_matrix[GENRES[label]][i] += 1
            confusion_matrix[GENRES[label]] /= len(preds)
        plt.title('confusion matrix for music recommendation model')
        plt.xlabel('labels')
        plt.ylabel('predictions')

        plt.savefig('models/16.1.5/confusion_matrix')
    return 1
