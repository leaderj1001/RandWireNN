import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable

from model import RandWire, SeparableConv2d

import numpy as np
import argparse

from torchviz import make_dot

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_data(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.ToTensor()), batch_size=args.batch_size
    )

    return train_loader, test_loader


class Model(nn.Module):
    def __init__(self, node_num, p, seed, in_channels, out_channels):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.seed = seed
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels // 2, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.rand_wire1 = nn.Sequential(
            RandWire(self.node_num, self.p, self.seed, self.in_channels, self.out_channels * 2)
        )
        self.rand_wire2 = nn.Sequential(
            RandWire(self.node_num, self.p, self.seed, self.in_channels * 2, self.out_channels * 2)
        )
        self.conv_output = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, 1280, kernel_size=1, stride=2),
            nn.BatchNorm2d(1280)
        )

        self.output = nn.Linear(1280, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.rand_wire1(x)
        x = self.rand_wire2(x)
        x = self.conv_output(x)

        # global average pooling
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = torch.squeeze(x)
        x = F.softmax(self.output(x), dim=-1)

        return x


def get_acc(model, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    return 100. * correct / len(test_loader.dataset)


def main():

    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs, (default: 50)')
    parser.add_argument('--p', type=float, default=0.4, help='graph probability, (default: 0.4)')
    parser.add_argument('--c', type=int, default=78, help='channel count for each node, (default: 78)')
    parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")
    parser.add_argument('--seed', type=int, default=42, help='seed, (default: 42)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate, (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size, (default: 64)')

    args = parser.parse_args()

    train_loader, test_loader = load_data(args)

    model = Model(args.node_num, args.p, args.seed, args.c, args.c).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        step = 0
        train_loss = []
        train_acc = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss.append(loss.data)
            optimizer.step()
            y_pred = output.data.max(1)[1]

            acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
            train_acc.append(acc)
            step += 1
            if step % 100 == 0:
                print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}".format(epoch, loss.data, acc))
        test_acc = get_acc(model, test_loader)
        print('Train set: Accuracy: {0:.3f}% Test set: Accuracy: {1:.2f}%'.format(np.sum(train_acc) / len(train_acc), test_acc))


if __name__ == '__main__':
    main()

