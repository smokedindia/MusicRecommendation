import torch.nn as nn


class MusicRecommendationModel(nn.Module):
    def __init__(self, n_bins, n_frames):
        super(MusicRecommendationModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=.5)
        )
        n_bins = n_bins // 2
        n_frames = n_frames // 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=.5)
        )
        n_bins = n_bins // 2
        n_frames = n_frames // 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=.5)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=128 * n_bins * n_frames, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(in_features=2048, out_features=1200),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(in_features=1200, out_features=10),
            nn.Softmax()
        )
        self.weights_init()

    def forward(self, out):
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)

        return out

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
