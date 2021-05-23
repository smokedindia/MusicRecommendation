import os
import shutil

import torch
import torch.nn as nn
import tqdm
from pydataclasses import DataClass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MusicRecommendationDataset, ToTensor
from model import MusicRecommendationModel


class TrainParams(DataClass):
    def __init__(self, train_config, **_extras):
        super(TrainParams, self).__init__(**_extras)
        self.num_epochs = train_config['num_epochs']
        self.batch_size = train_config['batch_size']
        self.learning_rate = train_config['learning_rate']


class Trainer:
    def __init__(self, config_list):
        train_params = TrainParams(config_list.train_config)
        feature_version = '.'.join([config_list.dataset_config['version'],
                                    config_list.feature_config['version']])
        self.data_root = 'dataset_feature/version_' + feature_version
        self.model_type = config_list.feature_config['feature_type']
        self.num_epochs = train_params.num_epochs
        self.batch_size = train_params.batch_size
        self.learning_rate = train_params.learning_rate

        self.train_ver = config_list.train_config['version']
        self.dataset_ver = config_list.dataset_config['version']
        self.feature_ver = config_list.feature_config['version']
        self.load_mem = False if config_list.feature_config[
                                     'feature_type'] == 'stft' else True

    def create_loaders(self):
        """

        creates dataloaders for training, validation, and test process

        Returns:
            3 torch.utils.data.dataloader in train, validation, test order

        """
        transform = ToTensor()
        dataset = MusicRecommendationDataset(transform=transform,
                                             root=self.data_root,
                                             load_mem=self.load_mem)

        # data split ratio = 8 : 1 : 1 (train : validation : test)
        train_dataset, rest_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[round(.8 * len(dataset)),
                     round(.2 * len(dataset))], )
        len_val_dataset = len(rest_dataset) // 2
        len_test_dataset = len(rest_dataset) - len_val_dataset
        val_dataset, test_dataset = torch.utils.data.random_split(
            dataset=rest_dataset,
            lengths=[len_val_dataset, len_test_dataset], )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        return train_loader, val_loader, test_loader

    def fit(self):
        """

        performs training process
        training and validation values are recorded on tensorboard

        Returns: None

        """
        # create directory and tensorboard SummaryWriter for version
        version_all = '.'.join([self.dataset_ver,
                                self.feature_ver,
                                self.train_ver])
        writer_path = 'runs/%s' % version_all
        if os.path.isdir(writer_path):
            print('tensorboard version already exists. removing content')
            shutil.rmtree(writer_path)
        writer = SummaryWriter(writer_path)

        save_root = 'models/%s' % version_all
        if os.path.exists(save_root):
            print('train version already exists. removing content.')
            shutil.rmtree(save_root)
        os.makedirs(save_root)

        train_loader, val_loader, test_loader = self.create_loaders()
        dataloaders = {'train': train_loader, 'val': val_loader}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = MusicRecommendationModel()
        criterion = nn.CrossEntropyLoss()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.learning_rate, )

        prog_bar = tqdm.tqdm(desc='training in progress',
                             total=self.num_epochs,
                             position=0,
                             leave=True)
        for epoch in range(self.num_epochs):
            accuracies = {}
            losses = {}
            running_loss = 0.
            running_corr = 0
            num_items = 0
            for stage, dataloader in dataloaders.items():
                with torch.set_grad_enabled(stage == 'train'):
                    prog_bar2 = tqdm.tqdm(
                        desc='epoch %s %s' % (epoch + 1, stage),
                        total=len(dataloader),
                        position=1,
                        leave=True)
                    for idx, batch in enumerate(dataloader):
                        if stage == 'train':
                            model.train()
                            optimizer.zero_grad()
                        else:
                            model.eval()

                        data = batch.get('spectrogram').to(device)
                        labels = batch.get('label').to(device).unsqueeze(1)
                        preds = model(data)

                        loss = criterion(preds, labels)
                        running_loss += loss.item()

                        num_correct = (preds == labels.view(-1)).sum()
                        running_corr += num_correct
                        num_items += labels.shape[0]

                        if stage == 'train':
                            loss.backward()
                            optimizer.step()
                        prog_bar2.update()
                    prog_bar2.close()

                    accuracies[stage] = running_corr / num_items
                    running_corr = 0

                    losses[stage] = running_loss / num_items
                    num_items = 0

            # code for histogram drawing
            #
            # layers = [model.module.layer1, model.module.layer2,
            #           model.module.layer3, model.module.layer4]
            # linear_count = 1
            # for i in range(len(layers)):
            #     for layer in layers[i]:
            #         layer_name = layer._get_name()
            #         if layer_name == 'Conv2d':
            #             desc = 'layer %s - %s' % (i + 1, layer_name)
            #             writer.add_histogram(desc,
            #                                  layer.weight)
            #         elif layer_name == 'Linear':
            #             desc = 'layer %s - %s' % (i + 1,
            #                                       layer_name + str(
            #                                           linear_count))
            #             writer.add_histogram(desc,
            #                                  layer.weight)
            #             if linear_count == 1:
            #                 linear_count += 1
            #             else:
            #                 linear_count = 1

            writer.add_scalars('accuracy', accuracies,
                               global_step=epoch + 1)
            writer.add_scalars('loss', losses, global_step=epoch + 1)

            torch.save(model.state_dict(),
                       os.path.join(save_root, str(epoch + 1)))
            prog_bar.update()
        prog_bar.close()

        writer.close()
