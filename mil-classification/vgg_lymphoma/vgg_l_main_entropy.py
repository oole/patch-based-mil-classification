import vgg_lymphoma.vgg_l_data as ldata
import vgg_lymphoma.vgg_l_net as lnet
import os
import train
import train_em
import data_tf
import numpy as np

from notification.tgm import  tgClient

from patch_selection.disc_entropy import EctropyDiscFinder
from disc_patch_select.disc_original import OriginalDiscFinder


SPLIT_SEED = 1337

BATCH_SIZE = 256
SPATIALSMOOTHING = False
LEARNING_RATE = 0.0001
DROPOUT_RATIO = 0.5
THRESHOLD_PERCENTILE = 0.8
TRAIN_DISC_FINDER = OriginalDiscFinder
PRED_DISC_FINDER = EctropyDiscFinder(THRESHOLD_PERCENTILE)

NOTIFICATION_CLIENT = tgClient()


BASENAME = "vgg_lymph"
SIMPLERUNSTAMP = "190123_l-0.0001_drop-0.5_bs-256"
EMRUNSTAMP = SIMPLERUNSTAMP + "_entropy-" + str(THRESHOLD_PERCENTILE) + "-second"

'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def simple_training(numberOfEpochs = 2):
    initialEpoch = 0
    epochs=numberOfEpochs

    netRoot = "/home/oole/lymphoma_net_vgg/"
    runName = BASENAME + "_simple_" + SIMPLERUNSTAMP + "/"
    modelName = BASENAME + "_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    simple_train_savepath = netRoot + runName + BASENAME + "_simple"
    em_train_savepath = netRoot + runName + BASENAME + "_em"
    logfile_path = netRoot + runName + BASENAME + "_net_log.csv"
    logreg_savepath = netRoot + runName + BASENAME + "_logreg"


    # load data
    # split into train val
    basePath = "/home/oole/data_lymphoma/"
    trainDataPath = basePath + "train/"
    testDataPath = basePath + "test/"
    trainSlideData = ldata.collect_data(trainDataPath)
    testSlideData =  ldata.collect_data(testDataPath)

    train.train_net(trainSlideData, testSlideData,
                    num_epochs=epochs,
                    batch_size=BATCH_SIZE,
                    savepath = simple_train_savepath,
                    do_augment = True,
                    model_name=modelName,
                    getlabel_train=ldata.getlabel,
                    log_savepath=logreg_savepath,
                    runName=runName,
                    lr=LEARNING_RATE,
                    buildNet = lnet.getLymphNet,
                    valIsTestData=True,
                    splitSeed=SPLIT_SEED)

    print("Data collected.")

'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''

def continue_simple_training(initialEpoch, epochNumber):

    netRoot = "/home/oole/lymphoma_net_vgg/"
    runName = BASENAME + "_simple_" + SIMPLERUNSTAMP + "_cont/"
    modelName = BASENAME + "_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    old_simple_savepath = netRoot + runName + BASENAME + "_simple"
    simple_cont_savepath = netRoot + runName + modelName
    logfile_path = netRoot + runName + BASENAME + "_net_log_em.csv"
    logreg_savepath = netRoot + runName + BASENAME + "_logreg"

    # load data
    # split into train val
    basePath = "/home/oole/data_lymphoma/"
    trainDataPath = basePath + "train/"
    testDataPath = basePath + "test/"
    trainSlideData = ldata.collect_data(trainDataPath)
    testSlideData = ldata.collect_data(testDataPath)

    train.train_net(trainSlideData, testSlideData,
                    num_epochs=epochNumber, batch_size=BATCH_SIZE,
                    savepath=simple_cont_savepath, do_augment=True,
                    model_name=modelName, getlabel_train=ldata.getlabel, log_savepath=logreg_savepath,
                    runName=runName, lr=LEARNING_RATE,
                    buildNet=lnet.getLymphNet,
                    valIsTestData=True,
                    initialEpoch=initialEpoch,
                    loadpath=old_simple_savepath,
                    splitSeed=SPLIT_SEED)

    print("Data collected.")


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def em_training(initialEpoch = 2, epochNumber=198):

    netRoot = "/home/oole/lymphoma_net_vgg/"
    modelName = BASENAME + "_model"
    runName = BASENAME + "_em_" + EMRUNSTAMP + "/"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    simple_train_loadpath = netRoot + BASENAME + "_simple_" + SIMPLERUNSTAMP + "/" + BASENAME + "_simple"
    em_train_savepath = netRoot + runName + BASENAME + "_em"
    logfile_path = netRoot + runName + BASENAME + "_net_log_em.csv"
    logreg_savepath = netRoot + runName + BASENAME + "_logreg"


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
                     num_epochs=epochNumber, dropout_ratio=DROPOUT_RATIO, learning_rate=LEARNING_RATE, sanity_check=False,
                     logfile_path=logfile_path,
                     logreg_savepath=logreg_savepath,
                     runName=runName,
                     netAcc=None,
                     buildNet = lnet.getLymphNet,
                     valIsTestData=True,
                     discriminativePatchFinderTrain=TRAIN_DISC_FINDER,
                     discriminativePatchFinderPredict=PRED_DISC_FINDER,
                     splitSeed = SPLIT_SEED)

    print("Data collected.")


'''
This will compare the networks performance to the knime blogpost on lymphoma classification.

Dataset must be pre split into training/testing
'''
def continue_em_training(epochNumber):
    initialEpoch = 103

    netRoot = "/home/oole/lymphoma_net/"
    runName = BASENAME + "_em_" + EMRUNSTAMP + "_cont/"
    modelName = "lymph_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")
    if not os.path.exists(netRoot + runName):
        os.makedirs(netRoot + runName)
    else:
        print("Run folder already extists.")

    old_em_savepath = netRoot + runName + BASENAME + "_em"
    em_train_savepath = netRoot + runName + BASENAME + "_em"
    logfile_path = netRoot + runName + BASENAME + "_net_log_em.csv"
    logreg_savepath = netRoot + runName + BASENAME + "_logreg"


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
                     num_epochs=epochNumber,
                     dropout_ratio=DROPOUT_RATIO,
                     learning_rate=LEARNING_RATE,
                     sanity_check=False,
                     logfile_path=logfile_path,
                     logreg_savepath=logreg_savepath,
                     runName=runName,
                     netAcc=None,
                     buildNet = lnet.getLymphNet,
                     valIsTestData=True,
                     discriminativePatchFinderTrain=TRAIN_DISC_FINDER,
                     discriminativePatchFinderPredict=PRED_DISC_FINDER,
                     splitSeed=SPLIT_SEED)

    print("Data collected.")

isSimple = None
# isSimple =  True

#simple_training(2)

# continue_simple_training()

em_training(initialEpoch = 2, epochNumber=118)

# continue_em_training()

if (isSimple is not None and isSimple):
    NOTIFICATION_CLIENT.sendmsg("Training done: \n"
                            + SIMPLERUNSTAMP)
else:
    NOTIFICATION_CLIENT.sendmsg("Training done: \n"
                                + EMRUNSTAMP)