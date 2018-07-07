import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import tensorflow as tf

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


def write_log_file(logfilePath, epochNum="NA", trainPredictAccuracy="NA", trainMaxAccuracy="NA",
                   trainLogRegAcccuracy="NA", trainLoss="NA", trainAccuracy="NA", valLoss="NA",
                   valAccuracy="NA", valMaxAccuracy="NA", valLogRegAccuracy="NA"):
    if logfilePath is not None:
        with open(logfilePath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epochNum, trainPredictAccuracy, trainMaxAccuracy, trainLogRegAcccuracy, trainLoss, trainAccuracy,
                 valLoss, valAccuracy, valMaxAccuracy, valLogRegAccuracy])

def writeBatchStatsToTensorBoard(trainLoss, trainAccuracy, summaryWriter, step):
    writeScalarSummary(trainLoss, "trainLossBatch", summaryWriter, step)
    writeScalarSummary(trainAccuracy, "trainAccuracyBatch", summaryWriter, step)

def writeEpochStatsToTensorBoard(trainLoss, trainAccuracy, summaryWriter, step):
    writeScalarSummary(trainLoss, "trainLossEpoch", summaryWriter, step)
    writeScalarSummary(trainAccuracy, "trainAccuracyEpoch", summaryWriter, step)

def writeValStatsToTensorBoard(valLoss, valAcc, summaryWriter, step):
    writeScalarSummary(valLoss, "simpleValLoss", summaryWriter, step)
    writeScalarSummary(valAcc, "simpleValAcc", summaryWriter, step)


def writeScalarSummary(scalar, scalarName, summaryWriter,  step):
    summaryWriter.add_summary(tf.Summary(value=[tf.Summary.Value(tag=scalarName, simple_value=scalar)]), global_step=step)