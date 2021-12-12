import torch
from torch import nn

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.hidden_layer = nn.Linear(3, 8)
        self.conv = nn.Conv1d(8, 3)
        self.decoder = nn.Linear(3, 3)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.conv(x)
        x = self.decoder(x)

        return x

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class FusionModel(torch.nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.use_layer = nn.Sequential(nn.Linear(3, 8),
                                       nn.Dropout(p=0.2),
                                              nn.ReLU(),
                                              nn.Linear(8, 8),
                                       nn.Dropout(p=0.2),
                                              nn.ReLU(),
                                              nn.Linear(8, 8),
                                       nn.Dropout(p=0.2),
                                              nn.ReLU(),
                                              nn.Linear(8, 1),
                                       nn.Dropout(p=0.2)
                                              )

        self.text_encoder = nn.Sequential(nn.Linear(20, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 128))

        self.conv1 = nn.Sequential(nn.Conv1d(3, 8, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm1d(8),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=8, stride=1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())

        self.media_layer = nn.Sequential(nn.Linear(128, 8))

        self.last_layer = nn.Linear(9, 3)

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, user, aural, visual, text):
        user = self.use_layer(user)
        text = self.text_encoder(text)

        media = torch.stack([visual, aural, text], dim=1)
        media = self.conv1(media)
        media = self.conv2(media)
        media = self.conv3(media)
        media = self.conv4(media)
        media = self.conv5(media)

        media = torch.squeeze(media)
        media = self.dropout(self.media_layer(media))

        fusion = torch.concat([media, user], dim=1)
        output = self.dropout(self.last_layer(fusion))
        return output

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
