import vgg_lymphoma.vgg_l_data as ldata
import vgg_lymphoma.vgg_l_net as lnet
import os
import train
import train_em
import data_tf
import numpy as np


BATCH_SIZE = 256
SPATIALSMOOTHING = False
LEARNING_RATE = 0.005
DROPOUT_RATIO = 0.5


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def simple_training():
    epochs=100
    initialEpoch = 0

    netRoot = "/home/oole/lymphoma_net_vgg/"
    runName = "vgg_lymphoma_simple_180906_2/"
    modelName = "vgg_lymph_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    simple_train_savepath = netRoot + runName + "vgg_lymph_simple"
    em_train_savepath = netRoot + runName + "vgg_lympf_em"
    logfile_path = netRoot + runName + "vgg_lymph_net_log.csv"
    logreg_savepath = netRoot + runName + "vgg_lymph_logreg"


    # load data
    # split into train val
    basePath = "/home/oole/data_lymphoma/"
    trainDataPath = basePath + "train/"
    testDataPath = basePath + "test/"
    trainSlideData = ldata.collect_data(trainDataPath)
    testSlideData =  ldata.collect_data(testDataPath)

    train.train_net(trainSlideData, testSlideData,
                    num_epochs=epochs, batch_size=BATCH_SIZE,
                    savepath = simple_train_savepath, do_augment = True,
                    model_name=modelName, getlabel_train=ldata.getlabel, log_savepath=logreg_savepath,
                    runName=runName, lr=LEARNING_RATE,
                    buildNet = lnet.getLymphNet, valIsTestData=True)

    print("Data collected.")

'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''

def continue_simple_training():
    epochs = 100
    initialEpoch = 100

    netRoot = "/home/oole/lymphoma_net_vgg/"
    runName = "vgg_lymphoma_simple_180902_cont/"
    modelName = "vgg_lymph_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    old_simple_savepath = netRoot + "vgg_lymphoma_simple_180902/" + "vgg_lymph_simple"
    simple_cont_savepath = netRoot + runName + modelName
    logfile_path = netRoot + runName + "vgg_lymph_net_log_em.csv"
    logreg_savepath = netRoot + runName + "vgg_lymph_logreg"

    # load data
    # split into train val
    basePath = "/home/oole/data_lymphoma/"
    trainDataPath = basePath + "train/"
    testDataPath = basePath + "test/"
    trainSlideData = ldata.collect_data(trainDataPath)
    testSlideData = ldata.collect_data(testDataPath)

    train.train_net(trainSlideData, testSlideData,
                    num_epochs=epochs, batch_size=BATCH_SIZE,
                    savepath=simple_cont_savepath, do_augment=True,
                    model_name=modelName, getlabel_train=ldata.getlabel, log_savepath=logreg_savepath,
                    runName=runName, lr=LEARNING_RATE,
                    buildNet=lnet.getLymphNet,
                    valIsTestData=True,
                    initialEpoch=initialEpoch,
                    loadpath=old_simple_savepath)

    print("Data collected.")


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def em_training():
    initialEpoch = 100

    netRoot = "/home/oole/lymphoma_net_vgg/"
    modelName = "vgg_lymph_model"
    runName = "vgg_lymphoma_em_180903_em/"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    simple_train_loadpath = netRoot + "vgg_lymphoma_simple_180902/" + "vgg_lymph_simple"
    em_train_savepath = netRoot + runName + "vgg_lympf_em"
    logfile_path = netRoot + runName + "vgg_lymph_net_log_em.csv"
    logreg_savepath = netRoot + runName + "vgg_lymph_logreg"


    # load data
    # split into train val
    basePath = "/home/oole/data_lymphoma/"
    trainDataPath = basePath + "train/"
    testDataPath = basePath + "test/"
    trainSlideData = ldata.collect_data(trainDataPath)
    testSlideData = ldata.collect_data(testDataPath)


    train_em.emtrain(trainSlideData, testSlideData,
                     simple_train_loadpath, em_train_savepath, BATCH_SIZE,
                     initial_epochnum=initialEpoch,
                     model_name=modelName,
                     spatial_smoothing=SPATIALSMOOTHING,
                     do_augment=True,
                     num_epochs=100, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE, sanity_check=False,
                     logfile_path=logfile_path,
                     logreg_savepath=logreg_savepath,
                     runName=runName,
                     netAcc=None,
                     buildNet = lnet.getLymphNet,
                     valIsTestData=True)

    print("Data collected.")


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def continue_em_training():
    initialEpoch = 103

    netRoot = "/home/oole/lymphoma_net/"
    runName = "lymphoma_em_180815_2_cont/"
    modelName = "lymph_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    old_em_savepath = netRoot + "lymphoma_em_180815_2/" + "lympf_em"
    em_train_savepath = netRoot + runName + "lympf_em"
    logfile_path = netRoot + runName + "lymph_net_log_em.csv"
    logreg_savepath = netRoot + runName + "lymph_logreg"


    # load data
    # split into train val
    dataPath = "/home/oole/data_lymphoma/"
    trainSlideData = ldata.collect_data(dataPath)
    valSlideData = data_tf.SlideData(trainSlideData.getSlideList(),
                                     None, np.asarray(trainSlideData.getSlideList()).size,
                                     trainSlideData.getSlideLabelList(),
                                     trainSlideData.getLabelFunc(),
                                     False,
                                     labelencoder=trainSlideData.getLabelEncoder(), parseFunctionAugment=trainSlideData.getparseFunctionNormal(), parseFunction=trainSlideData.getparseFunctionNormal())


    train_em.emtrain(trainSlideData, valSlideData,
                     old_em_savepath, em_train_savepath, BATCH_SIZE,
                     initial_epochnum=initialEpoch,
                     model_name=modelName,
                     spatial_smoothing=SPATIALSMOOTHING,
                     do_augment=True,
                     num_epochs=100, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE, sanity_check=False,
                     logfile_path=logfile_path, logreg_savepath=logreg_savepath, runName=runName, netAcc=None, buildNet = lnet.getLymphNet)

    print("Data collected.")

simple_training()

# continue_simple_training()

# em_training()

# continue_em_training()
