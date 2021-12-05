import torch
from torch import nn

class SimpleModel(torch.Module):
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
