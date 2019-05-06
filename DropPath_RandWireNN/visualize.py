import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def img_show(img, time_interval=2, img_save=False):
    plt.matshow(img)
    plt.show(block=False)
    plt.pause(time_interval)
    plt.close()


def heatmap_show(img, width, height, time_interval=2, img_save=False):
    heatmap = np.mean((img.data).cpu().numpy(), axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (width, height))
    plt.matshow(heatmap)
    plt.show(block=False)
    plt.pause(time_interval)
    plt.close()


def RGB_heatmap_show(img, original, width, height, time_interval=2, img_save=False):
    heatmap = np.mean((img.data).cpu().numpy(), axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * heatmap) # resize를 RGB 포맷으로 변환함.
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # heatmap으로 변환함.
    original = np.reshape(original.data.cpu().numpy(), [height, width, 3])
    combine_img = heatmap * 0.4 + original
    print(np.array(combine_img).shape)
    plt.matshow(combine_img[:, :, 0])
    plt.show(block=False)
    plt.pause(time_interval)
    plt.close()


def just_resize_show(img, width, height, time_interval=2, img_save=False):

    heatmap = cv2.resize((img.data).cpu().numpy(), (width, height))
    plt.matshow(heatmap)
    plt.show(block=False)
    plt.pause(time_interval)
    plt.close()


def layer_visualize(data, conv2_output, conv3_output, conv4_output, epoch):
    fig = plt.figure(figsize=(15, 6))
    for i in range(1, 5):
        subplot = fig.add_subplot(1, 4, i)
        if i == 1:
            resize = np.reshape(data.data.cpu().numpy(), [3, 32, 32])
            resize = np.mean(resize, axis=0)
        elif i == 2:
            resize = np.mean(conv2_output.data.cpu().numpy(), axis=0)
            resize = cv2.resize(resize, (32, 32))
        elif i == 3:
            resize = np.mean(conv3_output.data.cpu().numpy(), axis=0)
            resize = cv2.resize(resize, (32, 32))
        elif i == 4:
            resize = np.mean(conv4_output.data.cpu().numpy(), axis=0)
            resize = cv2.resize(resize, (32, 32))
        subplot.imshow(resize)
    if not os.path.isdir('visualize_img'):
        os.mkdir('visualize_img')
    plt.savefig("./visualize_img/" + str(epoch) + "_visualize.png")
    plt.close()


# weight 가져오는 부분
# print(list(model.conv1_2.weight))

# 여러 이미지를 한꺼번에 보여주는 코드
# fig = plt.figure(figsize=(15, 6))
# for i in range(10):
#     subplot = fig.add_subplot(2, 5, i + 1)
#     subplot.set_xticks([])
#     subplot.set_yticks([])
#
#     if i < 5:
#         subplot.imshow(np.array(data[i]).reshape((28, 28)))
#     else:
#         resize = cv2.resize((p[i - 5].data).cpu().numpy(), (28, 28))
#         subplot.imshow(resize)
# plt.show(block=False)
# plt.pause(1)
# plt.close()