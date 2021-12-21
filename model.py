import torch
from torch import nn

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.hidden_layer_0 = nn.Linear(279, 64)
        self.hidden_layer_1 = nn.Linear(64, 16)
        self.decoder = nn.Linear(16, 3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, user, aural, visual, text):
        x = torch.cat([aural, visual, text, user], dim=1)
        x = self.hidden_layer_0(x)
        x = self.hidden_layer_1(x)
        x = self.decoder(x)
        return x

        return a

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MLPFusionModel(nn.Module):

    def __init__(self):
        super(MLPFusionModel, self).__init__()
        self.user_encoder = nn.Linear(3, 5)
        self.media_encoder = nn.Linear(276, 20)
        self.drop_out = nn.Dropout(p=0.5)
        self.decoder = nn.Linear(25, 3)

    def forward(self, user, aural, visual, text):
        user = self.user_encoder(user)
        media = torch.cat([aural, visual, text], dim=1)
        media = self.media_encoder(media)
        # media = self.drop_out(media)

        x = torch.cat([user, media], dim=1)
        x = self.decoder(x)
        # x = self.drop_out(x)
        return x

    def init_weights(self, pretrained='', ):
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
                                              nn.ReLU(),
                                              nn.Linear(8, 8),
                                              nn.ReLU(),
                                              nn.Linear(8, 8),
                                              nn.ReLU(),
                                              nn.Linear(8, 1)
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


class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        input_dim: 每个输入xi的维度
        hidden_dim: 词向量嵌入变换的维度，也就是W的行数
        layer_dim: RNN神经元的层数
        output_dim: 最后线性变换后词向量的维度
        """
        super(RNNimc, self).__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim,
            batch_first=True,
            nonlinearity="relu"
        )

        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        维度说明：
            time_step = sum(像素数) / input_dim
            x : [batch, time_step, input_dim]
        """
        out, h_n = self.rnn(x, None)  # None表示h0会以全0初始化，及初始记忆量为0
        """
        out : [batch, time_step, hidden_dim]
        """
        out = self.fc1(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个h。
        return out