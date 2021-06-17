import argparse
import json
import os

from dataclasses import dataclass


DATASET_CONFIG_FILE = 'dataset_config.json'
TRAIN_CONFIG_FILE = 'train_config.json'
FEATURE_CONFIG_FILE = 'feature_config.json'
TEST_CONFIG_FILE = 'test_config.json'
api = None
audio_name = None

GENRES = {'blues': 0, 'classical': 1, 'country': 2,
          'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
          'pop': 7, 'reggae': 8, 'rock': 9}


@dataclass
class Configs:
    def __init__(self, config_root, versions):
        self.dataset_config = load_config(
            os.path.join(config_root, DATASET_CONFIG_FILE), versions[0])
        try:
            self.feature_config = load_config(
                os.path.join(config_root, FEATURE_CONFIG_FILE), versions[1])
            self.train_config = load_config(
                os.path.join(config_root, TRAIN_CONFIG_FILE), versions[2])
            if len(versions) > 3:
                self.test_version = versions[3]
        except IOError:
            pass
        self.version_all = '.'.join(versions[:-1])


def load_config(config_path, version):
    """

    loads configuration file and returns only designated version

    Args:
        config_path: str
            configuration file path that is either
            data generation, feature extraction, model, or train config path
        version: str
            version specification
            String type in order to use as key in configuration dictionary

    Returns:
        dictionary corresponding to the specified version

    """
    with open(config_path, 'rb') as f:
        config_total = json.load(f)
    result = config_total[version]
    # if default roots are specified in configuration file, update config
    # dictionary
    if config_total.get('roots') is not None:
        result.update(config_total['roots'])
    if config_total.get('links') is not None:
        result.update(config_total['links'])
    result['version'] = version
    return result


def create_data(config_list, save: bool = True):
    """performs .wav data creation process"""
    from gen_data import DatasetGenerator
    gen_data = DatasetGenerator(config_list.dataset_config, save=save)
    return gen_data.gen_dataset()


def extract_feature(config_list, raw_meta=None):
    """performs feature extraction process"""
    from preprocessing import FeatureExtractor
    extractor = FeatureExtractor(config_list, raw_meta=raw_meta)
    return extractor.extract()


def train(config_list, feature_meta=None, save_model=False):
    """performs training process"""
    from train import Trainer
    trainer = Trainer(config_list, feature_meta=feature_meta,
                      save_model=save_model)
    trainer.fit()


def execute(config_list):
    call_api(config_list)


def parse_ver(version_raw):
    """

    parses period split values of different versions and returns dataclass
    specifying version types

    Args:
        version_raw: str
            period split values, e.g. 0.0.0.0

    Returns: dataclass Versions
        list  of str versions for each modes

    """
    version_list = version_raw.split('.')
    if len(version_list) > 4:
        raise ValueError(
            'Invalid version format, up to 4 versions required: '
            'dataset.feature.train.test')
    return version_list


def call_api(config_list: Configs):
    from execution import get_prediction
    """
    specifies calls to API.

    Multithreading is a bit messy.
    The control flow goes into thread t specified
    

    """
    from ui import UI
    import threading
    l = threading.Lock()
    global api
    global audio_name

    def store_name(s: str):
        # s is the filename of .wav file
        global audio_name
        audio_name = s
        l.release()
        return

    def exit_handler():
        exit()

    api = UI(h=store_name, lock=l)  # h is called when user trims

    def get_model_prediction():
        global audio_name
        global api
        l.acquire()

        if audio_name is not None:
            """
            Call the model in here.
            I know, not the best isolation, but could not
            find a better solution for asynchronous problem
            """
            prediction = get_prediction(audio_name, config_list=config_list)

            api.setPrediction(prediction)

        l.release()

    t = threading.Thread(target=get_model_prediction)
    l.acquire()
    t.start()
    api.runApp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', type=str,
                   choices=['train', 'data', 'feature', 'test',
                            'all', 'exec'],
                   default='exec')
    # default version structure: dataset_version.feature_version.train_version
    p.add_argument('-v', '--version', type=str, default='16.1.5.0')

    p.add_argument('--config_root', type=str, default='./assets')
    p.add_argument('-n', '--norm', type=bool, default=False)
    p.add_argument('-s', '--save', type=bool, default=False)
    p.add_argument('--save_model', type=bool, default=False)
    args = p.parse_args()

    parsed_ver = parse_ver(args.version)
    config_list = Configs(config_root=args.config_root, versions=parsed_ver)
    if args.mode == 'data':
        create_data(config_list)
    elif args.mode == 'feature':
        extract_feature(config_list)
    elif args.mode == 'train':
        train(config_list)
    elif args.mode == 'exec':
        execute(config_list)
    else:  # if args.mode == 'all'
        raw_meta = create_data(config_list, args.save)
        feature_meta = extract_feature(config_list, raw_meta)
        train(config_list, feature_meta=feature_meta,
              save_model=args.save_model)


if __name__ == '__main__':
    main()
