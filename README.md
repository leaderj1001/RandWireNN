# Randomly Wired Neural Network
- Implement Exploring Randomly Wired Neural Networks for Image Recognition :)

![Datasets](https://img.shields.io/badge/Dataset-CIFAR--10-brightgreen.svg) ![Datasets](https://img.shields.io/badge/Dataset-CIFAR--100-green.svg) ![Datasets](https://img.shields.io/badge/Dataset-IMAGENET-yellowgreen.svg)

## Experiments
| Datasets | Model | Accuracy | Epoch | Training Time | Model Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | RandWireNN | 92.70% | 90 | 6h 17m | 9.27M
CIFAR-100 | RandWireNN | 71.43% | 50
CIFAR-100 | RandWireNN + Label Smoothing | 72.92% | 85
IMAGENET | WORK IN PROGRESS | WORK IN PROGRESS

## Update (2019.04.20)
1. I added graphing functions for train accuracy, test accuracy, and train loss.
2. I have added a part to report learning time and accuracy. Reporting of the above results can be seen in the reporting folder.
- Todo
  - I'll visualize the layer output.
  - I'll implement a "Drop Connection regularization".

### Plot
#### CIFAR-10
![epoch_acc_plot](https://user-images.githubusercontent.com/22078438/56430039-05cd4800-6300-11e9-8aa4-aac8038dbf9e.png)

#### CIFAR-100
![epoch_acc_plot](https://user-images.githubusercontent.com/22078438/56254892-8a03ad80-60fd-11e9-96c3-a0d25f980c6f.png)


## Run
```
python main.py
```
- If you want to change hyper-parameters, you can check "python main.py --help"

## Test
```
python test.py
```
- Put the saved model file in the checkpoint folder and type "python test.py".
- If you want to change hyper-parameters, you can check "python test.py --help"
- The model file currently in the checkpoint folder is a model with an accuracy of 92.70%.

## Reference
- [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/pdf/1904.01569.pdf)
  - Author: Saining Xie Alexander Kirillov Ross Girshick Kaiming He(Facebook AI Research, FAIR)
  - The paper is really awesome.
- [Random Graph Generator Module(networkx)](https://networkx.github.io/documentation/networkx-1.10/reference/generators.html)
- [Visualize Network in Pytorch](https://github.com/szagoruyko/pytorchviz)
- [Must have Module, cairosvg](https://cairosvg.org/)
- [Separable Convolution Code](https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py)
- [CIFAR benchmark](https://github.com/kuangliu/pytorch-cifar)
- [CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- [IMAGENET datasets](http://www.image-net.org/)
- Really thank you :)

## Methods
- Erdos-Renyi (ER) Graph, Watts-Strogatz (WS) Graph and Barabasi-Albert (BA) Graph are all available.
- If you want to visualize the network connection, you can follow the jupyter notebook in visualize_graph directory.
- Label smoothing.
  - In CIFAR-10, The accuracy was 92.00%.
  - But, CIFAR-100, I have seen improvements in CIFAR-100.

## Version
- Windows 10, Pycharm community...
- Python 3.7
- Cuda 9.2
- Cudnn 7.1.4
- pytorch 1.0.1
- networkx 2.2
- torchviz 0.0.1
- graphviz 0.10.1
- conda install cairo(If you want to visualize the network, it is a required module.)

## Todo
- Experiment with Imagenet dataset.
- To implement Optimzier like the paper.

# Network Image
- I have presented two graph visualizations. The ONNX module seems to be visualized more intuitively.
  - [ONNX module](https://github.com/leaderj1001/RandWireNN/tree/master/visualize_graph/ONNX_module)
  - [Graphviz module](https://github.com/leaderj1001/RandWireNN/tree/master/visualize_graph/graphviz_module)

## Small Network Image
- It is a picture of the sample small network in the visualize_graph directory.
- When I draw the contents of "Exploring Randomly Wired Neural Networks for Image Recognition" on the network, too many nodes are created. So I tried to draw a small network for visualization.
  - Number of nodes: 7
  - Graph parameters(probability P): 0.4
  - Random seed: 12
  - In_channels: 2
  - Out_channels: 2
- The following figure is a simple example, and the basic RandWired NeuralNetwork Module is provided.

## Example of Network (
![image](https://user-images.githubusercontent.com/22078438/55872389-d1eb7780-5bc7-11e9-95a6-7e053cefd1be.png)
