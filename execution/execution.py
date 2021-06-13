import os

import librosa
import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from main import Configs
from model import MusicRecommendationModel
from preprocessing import FeatureExtractor


def audio_to_tensors(user_input: np.ndarray,
                     config_list: Configs,
                     hop_size: float) -> torch.FloatTensor:
    """

    :param user_input: user input (trimmed audio data)
    :param config_list: dataclass for dataset, feature, train configuration
    :param hop_size: reference to slicing time
    :return:
            Tensor data for model input

    """
    # initialize required parameters from config file
    extractor = FeatureExtractor(config_list=config_list)
    time_res = int(extractor.feature_params.sr /
                   extractor.feature_params.feature_param.hop_length)
    feature_size = int(config_list.dataset_config['feature_size'] * time_res)
    hop_size = int(hop_size * time_res)
    pad_size = feature_size // 2

    y_feature = extractor.feature_func(user_input)
    y_feature_padded = np.pad(y_feature, (
        (0, 0), (feature_size - pad_size, feature_size)),
                              'constant', constant_values=0)
    n_input = int(y_feature.shape[1] // hop_size) + 1

    result = np.empty((n_input, y_feature.shape[0], feature_size))
    for i in range(n_input):
        hop_frame = i * hop_size
        spec_input = y_feature_padded[:, hop_frame: hop_frame + feature_size]
        scaler = StandardScaler()
        spec_input = scaler.fit_transform(spec_input)
        result[i, :, :] = spec_input

    return torch.from_numpy(result).type(torch.FloatTensor).to(
        'cuda').unsqueeze(1)


def divide_into_batches(model_input: torch.Tensor,
                        batch_size: int = 64) -> list:
    """
    divides input tensors to specified minibatch size to prevent
    memory overflow
    :param model_input: group of tensors for model input
    :param batch_size: batch size less than GPU overload
    :return:
        input batches similar to test dataset
    """
    batches = []
    num_batches = model_input.shape[0] // batch_size
    if model_input.shape[0] % batch_size:
        num_batches += 1
    for i in range(num_batches):
        batch = model_input[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)
    return batches


def get_prediction(user_input: str,
                   config_list: Configs,
                   hop_size: float = .5) -> int:
    model_path = os.path.join('models', config_list.version_all, 'model.ckpt')
    y, sr = librosa.load(user_input, sr=16000)
    model_input = audio_to_tensors(y, config_list, hop_size)
    model_input_batches = divide_into_batches(model_input)
    model_dims = model_input.shape

    state_dict = torch.load(model_path)
    model = torch.nn.DataParallel(
        MusicRecommendationModel(n_bins=model_dims[-2],
                                 n_frames=model_dims[-2])
    )
    model.load_state_dict(state_dict)

    predictions = np.array([])
    for batch in model_input_batches:
        preds_batch = model(batch).cpu().view(-1).detach().numpy().max()
        np.concatenate((predictions, preds_batch), axis=0)

    prediction = stats.mode(predictions)[0].max()
    return prediction
