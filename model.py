import torch
import torch.nn as nn
import torch.nn.functional as F

from randwire import RandWire, SeparableConv2d


class Model(nn.Module):
    def __init__(self, node_num, p, seed, in_channels, out_channels, graph_mode, model_mode):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.seed = seed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.model_mode = model_mode

        if self.model_mode is "CIFAR":
            self.CIFAR_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels),
            )
            self.CIFAR_conv2 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels, self.out_channels, self.graph_mode)
            )
            self.CIFAR_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels, self.out_channels * 2, self.graph_mode)
            )
            self.CIFAR_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels * 2, self.out_channels * 4, self.graph_mode)
            )
            self.CIFAR_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 4, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )
        elif self.model_mode is "SMALL":
            self.SMALL_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels // 2),
                nn.ReLU()
            )
            self.SMALL_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels // 2, out_channels=self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels)
            )
            self.SMALL_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels, self.out_channels, self.graph_mode)
            )
            self.SMALL_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels, self.out_channels * 2, self.graph_mode)
            )
            self.SMALL_conv5 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels * 2, self.out_channels * 4, self.graph_mode)
            )
            self.SMALL_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 4, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )
        elif self.model_mode is "REGULAR":
            self.REGULAR_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels // 2)
            )
            self.REGULAR_conv2 = nn.Sequential(
                RandWire(self.node_num // 2, self.p, self.seed, self.in_channels // 2, self.out_channels, self.graph_mode)
            )
            self.REGULAR_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels, self.out_channels * 2, self.graph_mode)
            )
            self.REGULAR_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels * 2, self.out_channels * 4, self.graph_mode)
            )
            self.REGULAR_conv5 = nn.Sequential(
                RandWire(self.node_num, self.p, self.seed, self.in_channels * 4, self.out_channels * 8, self.graph_mode)
            )
            self.REGULAR_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 8, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )

        self.output = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 10)
        )

    def forward(self, x):
        if self.model_mode is "CIFAR":
            out = self.CIFAR_conv1(x)
            out = self.CIFAR_conv2(out)
            out = self.CIFAR_conv3(out)
            out = self.CIFAR_conv4(out)
            out = self.CIFAR_classifier(out)
        elif self.model_mode is "SMALL":
            out = self.SMALL_conv1(x)
            out = self.SMALL_conv2(out)
            out = self.SMALL_conv3(out)
            out = self.SMALL_conv4(out)
            out = self.SMALL_conv5(out)
            out = self.SMALL_classifier(out)
        elif self.model_mode is "REGULAR":
            out = self.REGULAR_conv1(x)
            out = self.REGULAR_conv2(out)
            out = self.REGULAR_conv3(out)
            out = self.REGULAR_conv4(out)
            out = self.REGULAR_conv5(out)
            out = self.REGULAR_classifier(out)

        # global average pooling
        out = F.avg_pool2d(out, kernel_size=x.size()[2:])
        out = torch.squeeze(out)
        out = self.output(out)

        return out