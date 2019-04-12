# RandWiredNetwork
- Implement Exploring Randomly Wired Neural Networks for Image Recognition :)

## Current Cifar-10 datasets Result
Datasets | Accuracy
---------|----------
CIFAR-10 | 91.54%
---------|----------
CIFAR-100 | WORK IN PROGRESS
---------|----------
IMAGENET | WORK IN PROGRESS

![current](https://user-images.githubusercontent.com/22078438/56030401-b7480880-5d57-11e9-8387-03e761166b8a.PNG)

### Plot
![epoch_acc_plot](https://user-images.githubusercontent.com/22078438/56039705-d81c5800-5d6f-11e9-90c3-b6d588e55a70.png)


## Reference
- [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/pdf/1904.01569.pdf)
  - Author: Saining Xie Alexander Kirillov Ross Girshick Kaiming He(Facebook AI Research, FAIR)
  - The paper is really awesome.
- [Visualize Network in Pytorch](https://github.com/szagoruyko/pytorchviz)
- [Must have Module, cairosvg](https://cairosvg.org/)
- [Separable Convolution Code](https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py)
- [CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- [IMAGENET datasets](http://www.image-net.org/)
- Really thank you :)

## Methods
- Erdos-Renyi (ER) Graph, Watts-Strogatz (WS) Graph and Barabasi-Albert (BA) Graph are all available.
- If you want to visualize the network connection, you can follow the jupyter notebook in visualize_graph directory.

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
- Currently conducting experiments on Cifar-10 and Cifar-100 datasets.
- Currently there is no ImageNet dataset and it is downloading. Experiments will be conducted as data is downloaded. T.T
- To implement Optimzier like the paper.
- Label smoothing.

## Small Network Image
- It is a picture of the sample small network in the visualize_graph directory.
- When I draw the contents of "Exploring Randomly Wired Neural Networks for Image Recognition" on the network, too many nodes are created. So I tried to draw a small network for visualization.
  - Number of nodes: 7
  - Graph parameters(probability P): 0.4
  - Random seed: 12
  - In_channels: 2
  - Out_channels: 2
- The following figure is a simple example, and the basic RandWired NeuralNetwork Module is provided.

## Example of Network
![image](https://user-images.githubusercontent.com/22078438/55872389-d1eb7780-5bc7-11e9-95a6-7e053cefd1be.png)
