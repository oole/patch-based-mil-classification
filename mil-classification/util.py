import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

def update_print(msg):
    print(msg + "\r")


def plot_train_val_acc(train_acc, val_acc, title="Accuracy"):
    epoch_num = len(train_acc)
    plt.plot(range(0, epoch_num), np.asarray(train_acc), 'r')
    plt.plot(range(0, epoch_num), np.asarray(val_acc), 'b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    red_legend = mpatches.Patch(color='red', label='train acc')
    blue_legend = mpatches.Patch(color='blue', label='eval acc')
    plt.legend(handles=[red_legend, blue_legend])
    plt.title(title)
    plt.show()


def write_log_file(logfile_path, epochnum="", train_predict_accuracy="", train_max_accuracy="",
                   train_logreg_acccuracy="", train_accuracy="",
                   val_accuracy=""):
    if logfile_path is not None:
        with open(logfile_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epochnum, train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy, train_accuracy,
                 val_accuracy])