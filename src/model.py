import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class TinyTurnNet(nn.Module):
    def __init__(self, n_mels=64, conv_channels=(32, 64, 96), gru_hidden=96,
                 gru_layers=1, bidirectional=True, prosody_feat_dim=6,
                 num_classes=2, dropout=0.25):
        super().__init__()
        c1, c2, c3 = conv_channels

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv(c1, c2, stride=(2, 2))
        self.block2 = DepthwiseSeparableConv(c2, c3, stride=(2, 1))

        freq_bins_after = n_mels // 8
        gru_input_dim = c3 * freq_bins_after

        self.gru = nn.GRU(gru_input_dim, gru_hidden, num_layers=gru_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=0.0 if gru_layers == 1 else dropout)
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)

        self.prosody_proj = nn.Sequential(nn.Linear(prosody_feat_dim, 32), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim + 32, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, mel, prosody):
        x = self.stem(mel)
        x = self.block1(x)
        x = self.block2(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        _, h_n = self.gru(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.gru.bidirectional else h_n[-1]
        p = self.prosody_proj(prosody)
        fused = self.dropout(torch.cat([h, p], dim=-1))
        return self.classifier(fused)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
