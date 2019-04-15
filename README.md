# Randomly Wired Neural Network
- Implement Exploring Randomly Wired Neural Networks for Image Recognition :)

## Current Cifar-10 datasets Result
Datasets | Model | Accuracy | Epoch
----------|----------|----------|----------
CIFAR-10 | RandWireNN | 92.65% | 70
CIFAR-100 | RandWireNN | 71.43% | 50
IMAGENET | WORK IN PROGRESS | WORK IN PROGRESS

![current_acc](https://user-images.githubusercontent.com/22078438/56082204-d2e30a00-5e50-11e9-9b81-4caf98d70b92.PNG)

### Plot
![current_plot_5278](https://user-images.githubusercontent.com/22078438/56082202-c959a200-5e50-11e9-8e3f-57e60f706932.png)

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
- The model file currently in the checkpoint folder is a model with an accuracy of 92.65%.

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
