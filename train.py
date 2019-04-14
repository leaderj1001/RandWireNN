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


def draw_plot(epoch_list, train_loss_list, train_acc_list, val_acc_list):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(epoch_list, train_loss_list, label='training loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epoch_list, train_acc_list, label='train acc')
    plt.plot(epoch_list, val_acc_list, label='validation acc')
    plt.legend()

    if os.path.isdir('./plot'):
        plt.savefig('./plot/epoch_acc_plot.png')

    else:
        os.makedirs('./plot')
        plt.savefig('./plot/epoch_acc_plot.png')
    plt.close()


# reference
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
def load_data(args):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform_train), batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test), batch_size=args.batch_size, shuffle=False
    )

    return train_loader, test_loader


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
                nn.ReLU()
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
            out = self.REGULAR_conv1(out)
            out = self.REGULAR_conv1(out)
            out = self.REGULAR_conv1(out)
            out = self.REGULAR_conv1(out)
            out = self.REGULAR_classifier(out)

        # global average pooling
        out = F.avg_pool2d(out, kernel_size=x.size()[2:])
        out = torch.squeeze(out)
        out = self.output(out)

        return out


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in train_loader:
        adjust_learning_rate(optimizer, epoch, args)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print("  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length


def test(model, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    return 100. * float(correct) / len(test_loader.dataset)


def get_random_int():
    import random
    return random.randint(0, 10000)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    seed = get_random_int()
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=150, help='number of epochs, (default: 50)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.4)')
    parser.add_argument('--c', type=int, default=154, help='channel count for each node, 109, 154 (default: 154)')
    parser.add_argument('--k', type=int, default=4, help='Each node is connected to k nearest neighbors in ring topology, (Default: 4)')
    parser.add_argument('--m', type=int, default=5, help='Number of edges to attach from a new node to existing nodes, (Default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS", help="random graph, (Exampple: ER, WS, BA) default: WS")
    parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")
    parser.add_argument('--seed', type=int, default=seed, help="Random seed")
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning rate, (default: 1e-2)')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size, (default: 100)')
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--model-mode', type=str, default="CIFAR", help='CIFAR, SMALL, REGULAR, (default: CIFAR)')

    args = parser.parse_args()

    train_loader, test_loader = load_data(args)

    if args.load_model:
        model = Model(args.node_num, args.p, args.seed, args.c, args.c, args.graph_mode, args.model_mode).to(device)
        checkpoint = torch.load('./checkpoint/' + str(args.seed) + 'ckpt.t7')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
    else:
        model = Model(args.node_num, args.p, args.seed, args.c, args.c, args.graph_mode, args.model_mode).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        epoch_list.append(epoch)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args)
        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print('Test set accuracy: {0:.3f}%'.format(test_acc))

        if max_test_acc < test_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + str(args.seed) + 'ckpt.t7')
            max_test_acc = test_acc
            draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)


if __name__ == '__main__':
    main()


