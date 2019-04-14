import torch
from torchvision import datasets, transforms

from model import Model

import argparse
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def load_data(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test), batch_size=args.batch_size, shuffle=False
    )

    return test_loader


def main():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--seed', type=int, default=5278, help="seed, (default: 5278)")
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=154, help='channel count for each node, 109, 154 (default: 154)')
    parser.add_argument('--k', type=int, default=4, help='Each node is connected to k nearest neighbors in ring topology, (Default: 4)')
    parser.add_argument('--m', type=int, default=5, help='Number of edges to attach from a new node to existing nodes, (Default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS", help="random graph, (Exampple: ER, WS, BA) default: WS")
    parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")
    parser.add_argument('--model-mode', type=str, default="CIFAR", help='CIFAR, SMALL, REGULAR, (default: CIFAR)')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size, (default: 100)')

    args = parser.parse_args()

    test_loader = load_data(args)

    if os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"):
        model = Model(args.node_num, args.p, args.seed, args.c, args.c, args.graph_mode, args.model_mode).to(device)

        checkpoint = torch.load("./checkpoint/" + str(args.seed) + "ckpt.t7")
        model.load_state_dict(checkpoint['model'])
        end_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        print("[Saved Best Accuracy]: ", best_acc, '%', "[End epochs]: ", end_epoch)

        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(target.data).sum()
        print("[Test Accuracy] {0:2.2f}%".format(100. * float(correct) / len(test_loader.dataset)))
    else:
        assert os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"), "File not found. Please check again."


if __name__ == "__main__":
    main()
