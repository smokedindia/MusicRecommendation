import os
import shutil

import torch
import torch.nn as nn
import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MusicRecommendationDataset, ToTensor
from model import MusicRecommendationModel


@dataclass()
class TrainParams:
    def __init__(self, train_config, **_extras):
        super(TrainParams, self).__init__(**_extras)
        self.num_epochs = train_config['num_epochs']
        self.batch_size = train_config['batch_size']
        self.learning_rate = train_config['learning_rate']
        self.weight_decay = train_config.get('weight_decay', 0)


class Trainer:
    def __init__(self, config_list, feature_meta=None, save_model=False):
        train_params = TrainParams(config_list.train_config)
        feature_version = '.'.join([config_list.dataset_config['version'],
                                    config_list.feature_config['version']])
        self.data_root = 'dataset_feature/version_' + feature_version
        self.model_type = config_list.feature_config['feature_type']
        self.num_epochs = train_params.num_epochs
        self.batch_size = train_params.batch_size
        self.learning_rate = train_params.learning_rate
        self.weight_decay = train_params.weight_decay

        self.train_ver = config_list.train_config['version']
        self.dataset_ver = config_list.dataset_config['version']
        self.feature_ver = config_list.feature_config['version']
        self.feature_meta = feature_meta
        self.save_model = save_model

        self.genres = {0: 'blues', 1: 'classical', 2: 'country',
                       3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal',
                       7: 'pop', 8: 'reggae', 9: 'rock'}

    def create_loaders(self):
        """

        creates dataloaders for training, validation, and test process

        Returns:
            3 torch.utils.data.dataloader in train, validation, test order

        """
        transform = ToTensor()
        dataset = MusicRecommendationDataset(transform=transform,
                                             root=self.data_root,
                                             feature_meta=self.feature_meta)

        # data split ratio = 8 : 2 (train: validation)
        train_dataset, rest_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[round(.8 * len(dataset)),
                     round(.2 * len(dataset))], )
        val_dataset, test_dataset = torch.utils.data.random_split(
            dataset=rest_dataset,
            lengths=[round(len(rest_dataset) * .5),
                     len(rest_dataset) - round(len(rest_dataset) * .5)]
        )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=4)
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
            print('train version already exists. removing content.')
            shutil.rmtree(writer_path)
        writer = SummaryWriter(writer_path)

        train_loader, val_loader, test_loader = self.create_loaders()
        dataloaders = {'train': train_loader, 'val': val_loader}

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('Warning: Cuda device not found. training with cpu')

        # retrieve batch input dimensions for model initialization
        sample_batch = next(iter(val_loader))
        dim = sample_batch.get('spectrogram').shape
        writer.add_image('sample training data',
                         sample_batch['spectrogram'][0],
                         0)
        model = MusicRecommendationModel(n_bins=dim[-2], n_frames=dim[-1])
        criterion = nn.CrossEntropyLoss()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        optimizer = torch.optim.Adadelta(model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)

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
                    # prog_bar2 = tqdm.tqdm(
                    #     desc='epoch %s %s' % (epoch + 1, stage),
                    #     total=len(dataloader)
                    # )
                    for idx, batch in enumerate(dataloader):
                        if stage == 'train':
                            model.train()
                            optimizer.zero_grad()
                        else:
                            model.eval()

                        data = batch.get('spectrogram').to(device)
                        labels = batch.get('label').to(device)
                        preds = model(data)

                        loss = criterion(preds, labels)
                        running_loss += loss.item()

                        preds = torch.max(torch.round(preds), 1)[1].to(device)
                        num_correct = (preds == labels).sum()
                        running_corr += num_correct
                        num_items += labels.shape[0]

                        if stage == 'train':
                            loss.backward()
                            optimizer.step()
                    #     prog_bar2.update()
                    # prog_bar2.close()

                    accuracies[stage] = running_corr / num_items
                    running_corr = 0

                    losses[stage] = running_loss / num_items
                    num_items = 0

            layers = [model.module.layer1, model.module.layer2,
                      model.module.layer3, model.module.layer4]

            linear_count = None
            for i in range(len(layers)):
                if i == 0:
                    linear_count = 1
                for layer in layers[i]:
                    layer_name = layer._get_name()
                    if layer_name == 'Conv2d':
                        desc = 'layer %s - %s' % (i + 1, layer_name)
                        writer.add_histogram(desc,
                                             layer.weight,
                                             global_step=epoch + 1)
                    elif layer_name == 'Linear':
                        desc = 'layer %s - %s' % (i + 1,
                                                  layer_name + str(
                                                      linear_count)
                                                  )
                        writer.add_histogram(desc,
                                             layer.weight,
                                             global_step=epoch + 1)
                        linear_count += 1

            writer.add_scalars('accuracy', accuracies,
                               global_step=epoch + 1)
            writer.add_scalars('loss', losses, global_step=epoch + 1)
            prog_bar.update()
        prog_bar.close()
        writer.close()
        if self.save_model:
            save_root = f'models/{version_all}'
            os.makedirs(save_root)
            torch.save(model.state_dict(),
                       os.path.join(save_root, 'model.ckpt'))

        # 1. gets the probability predictions in a test_size x num_classes
        # Tensor 2. gets the preds in a test_size Tensor takes ~10 seconds
        # to run
        # class_probs = []
        # class_label = []
        # with torch.no_grad():
        #     for batch in test_loader:
        #         spectrogram = batch.get('spectrogram').to(device)
        #         labels = batch.get('label').to(device)
        #         output = model(spectrogram)
        #         class_probs_batch = [nn.Softmax(el) for el in output]
        #
        #         class_probs.append(class_probs_batch)
        #         class_label.append(labels)
        #
        # test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        # test_label = torch.cat(class_label)
        #
        # # helper function
        # def add_pr_curve_tensorboard(class_index, test_probs, test_label,
        #                              global_step=0):
        #     """
        #     Takes in a "class_index" from 0 to 9 and plots the corresponding
        #     precision-recall curve
        #     """
        #     tensorboard_truth = test_label == class_index
        #     tensorboard_probs = test_probs[:, class_index]
        #
        #     writer.add_pr_curve(self.genres[class_index],
        #                         tensorboard_truth,
        #                         tensorboard_probs,
        #                         global_step=global_step)
        #     writer.close()
        #
        # # plot all the pr curves
        # for i in range(len(self.genres)):
        #     add_pr_curve_tensorboard(i, test_probs, test_label)
