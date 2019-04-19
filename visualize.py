import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    combine_img = heatmap * 0.4 + original
    plt.matshow(combine_img)
    plt.show(block=False)
    plt.pause(time_interval)
    plt.close()


def just_resize_show(img, width, height, time_interval=2, img_save=False):
    heatmap = cv2.resize((img.data).cpu().numpy(), (width, height))
    plt.matshow(heatmap)
    plt.show(block=False)
    plt.pause(time_interval)
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