import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_sizes, strides, paddings, batch_norm: bool = False):
        super(ConvBlock, self).__init__()
        self.do_batch_norm = batch_norm
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_sizes, strides, paddings)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, num_class, map_to_seq_hidden=64, rnn_hidden=256):
        super(CRNN, self).__init__()
        # CNN block
        self.cnn = nn.Sequential(
            ConvBlock(img_channel, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(256, 512, 3, 1, 1, batch_norm=True),
            ConvBlock(512, 512, 3, 1, 1, batch_norm=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(512, 512, 2, 1, 0)
        )
        # map CNN to sequence
        self.map2seq = nn.Linear(
            512 * (img_height // 16 - 1), map_to_seq_hidden)
        # RNN
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        # fully connected
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, x):
        # CNN block
        x = self.cnn(x)
        # reformat array
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)
        x = self.map2seq(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.dense(x)
        return x
