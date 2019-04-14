import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from graph import RandomGraph


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# reference, Thank you.
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


# ReLU-convolution-BN triplet
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Unit, self).__init__()

        self.unit = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.unit(x)


class Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels, stride=1):
        super(Node, self).__init__()
        self.in_degree = in_degree
        if len(self.in_degree) > 1:
            self.weights = nn.Parameter(torch.zeros(len(self.in_degree)), requires_grad=True)
        self.unit = Unit(in_channels, out_channels, stride=stride)

    def forward(self, *input):
        if len(self.in_degree) > 1:
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            out = self.unit(x)
        else:
            out = self.unit(input[0])
        return out


class RandWire(nn.Module):
    def __init__(self, node_num, p, seed, in_channels, out_channels, graph_mode):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.seed = seed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode

        # get graph nodes and in edges
        graph = RandomGraph(self.node_num, self.p, self.seed, graph_mode=self.graph_mode)
        self.nodes, self.in_edges = graph.get_graph_info()

        # define input Node
        self.module_list = nn.ModuleList([Node(self.in_edges[0], self.in_channels, self.out_channels, stride=2)])
        # define the rest Node
        self.module_list.extend([Node(self.in_edges[node], self.out_channels, self.out_channels) for node in self.nodes if node > 0])

    def forward(self, x):
        memory = {}

        # start vertex
        out = self.module_list[0].forward(x)
        memory[0] = out

        # the rest vertex
        for node in range(1, len(self.nodes) - 1):
            if len(self.in_edges[node]) > 1:
                out = self.module_list[node].forward(*[memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:
                out = self.module_list[node].forward(memory[self.in_edges[node][0]])
            memory[node] = out

        out = self.module_list[self.node_num + 1].forward(*[memory[in_vertex] for in_vertex in self.in_edges[self.node_num + 1]])

        # According to the paper, we create a node that connects the input and output of a random graph.
        # The input and output nodes do not perform convolution operations.
        # However, performance is better with a separable convolution on the output node.

        # out = memory[self.in_edges[self.node_num + 1][0]]
        # for in_vertex_index in range(1, len(self.in_edges[self.node_num + 1])):
        #     out += memory[self.in_edges[self.node_num + 1][in_vertex_index]]
        # out = out / len(self.in_edges[self.node_num + 1])
        return out
