import breasthistology.b_data as bdata
import breasthistology.b_net as bnet
import os
import train
import train_em
import data_tf
import numpy as np


BATCH_SIZE = 256
SPATIALSMOOTHING = False
LEARNING_RATE = 0.0001
DROPOUT_RATIO = 0.5


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def simple_training():
    epochs=2
    initialEpoch = 0

    netRoot = "/home/oole/breasthistology_net/"
    runName = "breasthistology_simple_testing/"
    modelName = "breasthistology_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    simple_train_savepath = netRoot + runName + "breasthistology_simple"
    em_train_savepath = netRoot + runName + "breasthistology_em"
    logfile_path = netRoot + runName + "breasthistology_net_log.csv"
    logreg_savepath = netRoot + runName + "breasthistology_logreg"


    # load data
    # split into train val
    trainDataPath = "/home/oole/breasthistology/training/"
    testDataPath = "/home/oole/breasthistology/testing/"
    trainSlideData = bdata.collect_data(trainDataPath)
    testSlideData = bdata.collect_data(testDataPath)

    train.train_net(trainSlideData, testSlideData,
                    num_epochs=epochs, batch_size=BATCH_SIZE,
                    savepath = simple_train_savepath, do_augment = True,
                    model_name=modelName, getlabel_train=bdata.getlabel, log_savepath=logreg_savepath,
                    runName=runName, lr=0.0001,
                    buildNet = bnet.getBreasthistoNet, valIsTestData=True)

    print("Data collected.")


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def em_training():
    initialEpoch = 2

    netRoot = "/home/oole/breasthistology_net/"
    runName = "breasthistology_em_180815_2/"
    modelName = "breasthistology_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    simple_train_loadpath = netRoot + "breasthistology_simple_180815/" + "breasthistology_simple"
    em_train_savepath = netRoot + runName + "breasthistology_em"
    logfile_path = netRoot + runName + "breasthistology_net_log_em.csv"
    logreg_savepath = netRoot + runName + "breasthistology_logreg"


    # load data
    # split into train val
    trainDataPath = "/home/oole/breasthistology/training/"
    testDataPath = "/home/oole/breasthistology/testing/"
    trainSlideData = bdata.collect_data(trainDataPath)
    testSlideData = bdata.collect_data(testDataPath)


    train_em.emtrain(trainSlideData, testSlideData,
                     simple_train_loadpath, em_train_savepath, BATCH_SIZE,
                     initial_epochnum=initialEpoch,
                     model_name=modelName,
                     spatial_smoothing=SPATIALSMOOTHING,
                     do_augment=True,
                     num_epochs=100, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE, sanity_check=False,
                     logfile_path=logfile_path, logreg_savepath=logreg_savepath, runName=runName, netAcc=None,
                     buildNet = bnet.getBreasthistoNet,
                     valIsTestData=True)

    print("Data collected.")


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def continue_em_training():
    initialEpoch = 103

    netRoot = "/home/oole/breasthistology_net/"
    runName = "lymphoma_em_180815_2_cont/"
    modelName = "breasthistology_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    old_em_savepath = netRoot + "breasthistology_em_180815_2/" + "breasthistology_em"
    em_train_savepath = netRoot + runName + "breasthistology_em"
    logfile_path = netRoot + runName + "breasthistology_net_log_em.csv"
    logreg_savepath = netRoot + runName + "breasthistology_logreg"


    # load data
    # split into train val
    trainDataPath = "/home/oole/breasthistology/training/"
    testDataPath = "/home/oole/breasthistology/testing/"
    trainSlideData = bdata.collect_data(trainDataPath)
    testSlideData = bdata.collect_data(testDataPath)


    train_em.emtrain(trainSlideData, testSlideData,
                     old_em_savepath, em_train_savepath, BATCH_SIZE,
                     initial_epochnum=initialEpoch,
                     model_name=modelName,
                     spatial_smoothing=SPATIALSMOOTHING,
                     do_augment=True,
                     num_epochs=100, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE, sanity_check=False,
                     logfile_path=logfile_path, logreg_savepath=logreg_savepath, runName=runName, netAcc=None, buildNet = bnet.getBreasthistoNet)

    print("Data collected.")

simple_training()

# em_training()

# continue_em_training()
