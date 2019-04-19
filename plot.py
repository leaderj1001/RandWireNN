import matplotlib.pyplot as plt
import os


def draw_plot(epoch_list, train_loss_list, train_acc_list, val_acc_list):
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
