# Graph Visualize using ONNX module

## Required
- When you using ONNX module, you have to use torchvision==0.2.1
- And when you using pooling layer in ONNX, you have to use fixed kernel_size.
  - [problems reference](https://github.com/pytorch/pytorch/issues/10942)
- If you keep both of these, you will not have any problems using it.
- [ONNX graph generator module](https://discuss.pytorch.org/t/print-autograd-graph/692/35)
- To use the ONNX module, I used the hiddenlayer module. I will attach the reference below.
  - [hiddenlayer module github repository](https://github.com/waleedka/hiddenlayer)

## Random graph generator
- When the random graph is generated, it is shown as below.
![graph](https://user-images.githubusercontent.com/22078438/56333670-1088bf80-61d0-11e9-81b5-381ad33d6c34.PNG)

## Generate network
![grpah_pdf-1](https://user-images.githubusercontent.com/22078438/56336981-be4e9b00-61dd-11e9-9fba-db7927852de2.png)

