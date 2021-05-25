import argparse
import json
import os

from dataclasses import dataclass

DATASET_CONFIG_FILE = 'dataset_config.json'
TRAIN_CONFIG_FILE = 'train_config.json'
FEATURE_CONFIG_FILE = 'feature_config.json'
TEST_CONFIG_FILE = 'test_config.json'


@dataclass()
class Configs:
    def __init__(self, config_root, versions, **_extras):
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
    result['version'] = version
    return result


def create_data(config_list):
    """performs .wav data creation process"""
    from gen_data import DatasetGenerator
    gen_data = DatasetGenerator(config_list.dataset_config)
    gen_data.gen_dataset()


def extract_feature(config_list):
    """performs feature extraction process"""
    from preprocessing import FeatureExtractor
    extractor = FeatureExtractor(config_list)
    extractor.extract()


def train(config_list):
    """performs training process"""
    from train import Trainer
    trainer = Trainer(config_list)
    trainer.fit()


def parse_ver(version_raw):
    """

    parses period split values of different versions and returns dataclass
    specifying version types

    Args:
        version_raw: str
            period split values, e.g. 0.0.0

    Returns: dataclass Versions
        list  of str versions for each modes

    """
    version_list = version_raw.split('.')
    if len(version_list) > 4:
        raise ValueError(
            'Invalid version format, up to 4 versions required: '
            'dataset.feature.train.test')
    return version_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', type=str,
                   choices=['train', 'data', 'feature', 'test',
                            'all'],
                   default='train')
    # default version structure: dataset_version.feature_version.train_version
    p.add_argument('-v', '--version', type=str, default='1.1.2.0')
    p.add_argument('--config_root', type=str, default='./assets')
    p.add_argument('--hop_time', type=float, default=.5)
    p.add_argument('-n', '--norm', type=bool, default=False)
    args = p.parse_args()

    parsed_ver = parse_ver(args.version)
    config_list = Configs(config_root=args.config_root, versions=parsed_ver)
    if args.mode == 'data':
        create_data(config_list)
    elif args.mode == 'feature':
        extract_feature(config_list)
    elif args.mode == 'train':
        train(config_list)
    else:  # if args.mode == 'all'
        create_data(config_list)
        extract_feature(config_list)
        train(config_list)


if __name__ == '__main__':
    main()
