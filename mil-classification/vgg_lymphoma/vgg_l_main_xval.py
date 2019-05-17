from numpy.distutils.system_info import numarray_info

import vgg_lymphoma.vgg_l_data as ldata
import vgg_lymphoma.vgg_l_net as lnet
import os
import train
import train_em
import tensorflow as tf
import data_tf

from patch_selection.disc_entropy import EctropyDiscFinder
from disc_patch_select.disc_original import OriginalDiscFinder

SPLIT_SEED = 1337

BATCH_SIZE = 256
SPATIALSMOOTHING = False
LEARNING_RATE = 0.0001
DROPOUT_RATIO = 0.5
THRESHOLD_PERCENTILE = 0.3
TRAIN_DISC_FINDER = OriginalDiscFinder
PRED_DISC_FINDER = EctropyDiscFinder(THRESHOLD_PERCENTILE)

BASENAME = "vgg_lymph_xval"
SIMPLERUNSTAMP = "190516_l-" + str(LEARNING_RATE) + "_drop-" + str(DROPOUT_RATIO) + "_bs-" + str(
    BATCH_SIZE) + "_entropy-" + str(THRESHOLD_PERCENTILE)

'''
Trains the two-layer model in one go, and performs cross-validation.
'''


def cross_val_training(foldToTrain, numberOfEpochs=30):
    initialEpoch = 0

    netRoot = "/home/oole/lymphoma_net_vgg/"
    modelName = BASENAME + "_model"

    if not os.path.exists(netRoot):
        os.makedirs(netRoot)
    else:
        print("Net root folder already extists.")

    # load data
    # split into train val
    basePath = "/home/oole/data_lymphoma_full/"
    trainDataPath = basePath + "train/"
    # testDataPath = basePath + "test/"
    trainSlideData = ldata.collect_data(trainDataPath)
    testSlideData = trainSlideData

    sess = tf.Session()

    netAcc = None

    foldNum = 1

    for trainSlideData, testSLideData in data_tf.KFoldSlideList(trainSlideData, testSlideData, numberOfSplits=5,
                                                                splitSeed=1337):
        print("Fold number: " + str(foldNum))
        if foldNum == foldToTrain:
            # train slide data should consist of all available training data
            # is split into test and training
            runName = BASENAME + "_simple_" + SIMPLERUNSTAMP + "_k-" + str(foldNum) + "-5" + "/"

            if not os.path.exists(netRoot + runName):
                os.makedirs(netRoot + runName)
            else:
                print("Run folder already extists.")

            train_savepath = netRoot + runName + BASENAME + "_k-" + str(foldNum) + "-5"
            logreg_savepath = netRoot + runName + BASENAME + "_logreg" + "_k-" + str(foldNum) + "-5"
            logfile_path = netRoot + runName + BASENAME + "_k-" + str(foldNum) + "-5" + "_net_log_em.csv"

            if netAcc is not None:
                netAcc.getSummmaryWriter(runName, sess.graph, forceNew=True)

            _, _, netAcc = train.train_net(trainSlideData, testSlideData,
                                           num_epochs=2,
                                           batch_size=BATCH_SIZE,
                                           savepath=train_savepath,
                                           do_augment=True,
                                           model_name=modelName,
                                           getlabel_train=ldata.getlabel,
                                           runName=runName,
                                           lr=LEARNING_RATE,
                                           buildNet=lnet.getLymphNet,
                                           valIsTestData=True,
                                           splitSeed=SPLIT_SEED,
                                           sess=sess,
                                           netAcc=netAcc,
                                           initialEpoch=initialEpoch,
                                           verbose = 1,
                                           do_simple_validation=False)

            print("Finished Simple Training")

            train_em.emtrain(trainSlideData, testSlideData,
                             None, train_savepath, BATCH_SIZE,
                             initial_epochnum=initialEpoch + 2,
                             model_name=modelName,
                             spatial_smoothing=SPATIALSMOOTHING,
                             do_augment=True,
                             num_epochs=initialEpoch + numberOfEpochs, dropout_ratio=DROPOUT_RATIO,
                             learning_rate=LEARNING_RATE, sanity_check=False,
                             logreg_savepath=logreg_savepath,
                             runName=runName,
                             netAcc=netAcc,
                             valIsTestData=True,
                             discriminativePatchFinderTrain=TRAIN_DISC_FINDER,
                             discriminativePatchFinderPredict=PRED_DISC_FINDER,
                             splitSeed=SPLIT_SEED,
                             sess=sess,
                             verbose=1,
                             do_simple_validation=False)

        foldNum = foldNum + 1

        # Reset weights for each iteration
        #sess.run(tf.global_variables_initializer())


print("Start KFold Crossval")
foldToTrain = 3
numberOfEpochs = 50
cross_val_training(foldToTrain, numberOfEpochs)
