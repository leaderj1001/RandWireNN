# Drop Connection regularization

## Usage
- The drop connection rate used by the authors in the paper is 0.1. Therefore, the default drop connection rate is 0.1.

```
python main.py
```
- If you want to change hyper-parameters, you can check "python main.py --help"

Options:
- `--epochs` (int) - number of epochs, (default: 100).
- `--p` (float) - graph probability, (default: 0.75).
- `--c` (int) - channel count for each node, (example: 78, 109, 154), (default: 78).
- `--k` (int) - each node is connected to k nearest neighbors in ring topology, (default: 4).
- `--m` (int) - number of edges to attach from a new node to existing nodes, (default: 5).
- `--graph-mode` (str) - kinds of random graph, (exampple: ER, WS, BA), (default: WS).
- `--node-num` (int) - number of graph node (default n=32).
- `--learning-rate` (float) - learning rate, (default: 1e-1).
- `--model-mode` (str) - which network you use, (example: CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME), (default: CIFAR10).
- `--batch-size` (int) - batch size, (default: 100).
- `--dataset-mode` (str) - which dataset you use, (example: CIFAR10, CIFAR100, MNIST), (default: CIFAR10).
- `--is-train` (bool) - True if training, False if test. (default: True).
- `--drop-path` (float) - regularization by disconnecting between random graphs. (default: 0.1).
- `--load-model` (bool) - (default: False).
