import numpy as np
import netutil

import tensorflow as tf
import dataset
import train
import data_tf
import gc

import train_logreg
import util
import evaluate
from disc_patch_select.disc_original import OriginalDiscFinder

shuffle_buffer_size = 2048


def emtrain(trainSlideData, valSlideData,
            loadpath, savepath, batch_size,
            initial_epochnum=0,
            model_name='model',
            spatial_smoothing=False,
            do_augment=True,
            num_epochs=2,
            dropout_ratio=0.5,
            learning_rate=0.0005,
            sanity_check=False,
            logfile_path=None,
            logreg_savepath=None,
            runName="",
            netAcc=None,
            buildNet=netutil.build_model,
            valIsTestData=False,
            discriminativePatchFinderTrain=OriginalDiscFinder,
            discriminativePatchFinderPredict=OriginalDiscFinder,
            splitSeed=None,
            sess=tf.Session(),
            verbose=2,
            do_simple_validation=True):
    iteration = 0

    epochnum = initial_epochnum
    # get_per_class_instances(trainSlideData, label_encoder)

    if not valIsTestData:
        trainSlideData, valSlideData = data_tf.splitSlideLists(trainSlideData, valSlideData, splitSeed)

    old_disc_patches = trainSlideData.getNumberOfPatches()
    ## create iterator
    if (netAcc is None):
        create_iterator_patch = dataset.slidelist_to_patchlist(trainSlideData.getSlideList())
        create_iterator_dataset = dataset.img_dataset_augment(create_iterator_patch,
                                                              batch_size=batch_size,
                                                              shuffle_buffer_size=shuffle_buffer_size,
                                                              shuffle=True,
                                                              getlabel=trainSlideData.getLabelFunc(),
                                                              labelEncoder=trainSlideData.getLabelEncoder(),
                                                              parseFunctionAugment=trainSlideData.getparseFunctionAugment())
        create_iterator_iter = create_iterator_dataset.make_initializable_iterator()
        # proxy_iterator_handle_ph= tf.placeholder(tf.string, shape=[])
        # proxy_iterator = tf.data.Iterator.from_string_handle(proxy_iterator_handle_ph, output_types=create_iterator_iter.output_types,
        #                                                      output_shapes=create_iterator_iter.output_shapes)
        # x, y = proxy_iterator.get_next()
        iterator_handle, iterator_access, proxy_iterator = dataset.proxy_iterator(sess, create_iterator_iter)
        x, y = proxy_iterator.get_next()
        netAcc = buildNet(model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True,
                          batchSize=batch_size)
        netAcc.setIteratorHandle(iterator_handle)

    # model saver
    saver = tf.train.Saver()
    if loadpath is not None:
        ########################
        # load model from disc
        saver.restore(sess, loadpath)

    netAcc.getSummmaryWriter(runName, sess.graph)

    discriminativePatchFinderTrain = discriminativePatchFinderTrain(trainSlideData,
                                                                    netAcc,
                                                                    spatial_smoothing,
                                                                    sess,
                                                                    dropout_ratio=dropout_ratio,
                                                                    sanity_check=sanity_check, do_augment=do_augment,
                                                                    logreg_model_savepath=logreg_savepath,
                                                                    epochnum=epochnum)

    while epochnum <= num_epochs + initial_epochnum:

        # Do not need H here
        H, disc_patches_new = discriminativePatchFinderTrain.find_discriminative_patches()
        print("Discriminative patches: " + repr(disc_patches_new) + ". Before: " + repr(old_disc_patches))

        # H = None
        # disc_patches_new = trainSlideData.getNumberOfPatches()

        gc.collect()

        train_accuracy, val_accuracy = train_on_discriminative_patches(trainSlideData,
                                                                       valSlideData,
                                                                       netAcc,
                                                                       H,
                                                                       epochnum,
                                                                       2,
                                                                       disc_patches_new,
                                                                       dropout_ratio=dropout_ratio,
                                                                       learning_rate=learning_rate,
                                                                       sess=sess,
                                                                       do_augment=do_augment,
                                                                       runName=runName,
                                                                       logregSavePath=logreg_savepath,
                                                                       discriminativePatchFinderPredict=discriminativePatchFinderPredict,
                                                                       verbose=verbose,
                                                                       do_simple_validation=do_simple_validation)

        epochnum += 2
        old_disc_patches = disc_patches_new
        iteration = iteration + 1

        util.write_log_file(logfile_path, epochNum=epochnum, trainPredictAccuracy="", trainMaxAccuracy="",
                            trainLogRegAcccuracy="", trainAccuracy=train_accuracy,
                            valAccuracy=val_accuracy)
        if savepath is not None:
            saver.save(sess, savepath)

        # print("Iteration done (breaking)")
        # # Since there is a memory leak problem at the moment the learning will not be looped
        # break


def train_on_discriminative_patches(trainSlideData,
                                    valSlideData,
                                    netAccess,
                                    H,
                                    initial_epochnum,
                                    num_epochs,
                                    num_patches,
                                    dropout_ratio,
                                    learning_rate,
                                    sess,
                                    do_augment=False,
                                    runName="",
                                    logregSavePath=None,
                                    discriminativePatchFinderPredict=None,
                                    verbose = 2,
                                    do_simple_validation=True):
    slideList = trainSlideData.getSlideList()
    batchSize = netAccess.getBatchSize()

    train_patches = dataset.slidelist_to_patchlist(slideList, H=H)
    # Important shuffle!!! (shuffle buffer may be too small to properly shuffle
    np.random.shuffle(train_patches)
    val_patches = dataset.slidelist_to_patchlist(valSlideData.getSlideList())
    if len(train_patches) != num_patches:
        raise Exception("H did not work correctly")

    if do_augment:
        train_dataset = dataset.img_dataset_augment(train_patches,
                                                    batch_size=batchSize,
                                                    shuffle_buffer_size=shuffle_buffer_size,
                                                    shuffle=True,
                                                    getlabel=trainSlideData.getLabelFunc(),
                                                    labelEncoder=trainSlideData.getLabelEncoder(),
                                                    parseFunctionAugment=trainSlideData.getparseFunctionAugment())
    else:
        train_dataset = dataset.img_dataset(train_patches, batch_size=batchSize,
                                            shuffle_buffer_size=shuffle_buffer_size, shuffle=True)

    val_dataset = dataset.img_dataset(val_patches, batch_size=batchSize,
                                      getlabel=valSlideData.getLabelFunc(),
                                      labelEncoder=valSlideData.getLabelEncoder(),
                                      parseFunction=valSlideData.getparseFunctionNormal())
    val_iterator = val_dataset.make_initializable_iterator()
    train_iterator = train_dataset.make_initializable_iterator()
    actualEpoch = initial_epochnum
    train_accuracy = train.train_given_net(netAccess,
                                           len(train_patches),
                                           train_iterator,
                                           val_iterator_len=len(val_patches),
                                           val_iterator=val_iterator,
                                           num_epochs=num_epochs,
                                           batch_size=batchSize,
                                           dropout_ratio=dropout_ratio,
                                           learning_rate=learning_rate,
                                           sess=sess,
                                           runName=runName,
                                           actualEpoch=actualEpoch,
                                           verbose=verbose,
                                           do_simple_validation=do_simple_validation)

    # Train logreg model with current net
    logregModel = train_logreg.train_logreg(netAccess, logregSavePath, trainSlideData, dropout_ratio, sess,
                                            discriminativePatchFinderPredict)

    evaluate.evaluateNet(netAccess, logregModel, valSlideData, actualEpoch, sess=sess, dropout=dropout_ratio,
                         runName=runName, discriminativePatchFinder=discriminativePatchFinderPredict)
    actualEpoch += num_epochs

    return train_accuracy


def get_patch_number(slides):
    patchnum = 0
    for slide in slides:
        patchnum += len(slide)
    return patchnum


def get_per_class_instances(trainSlideData, LE):
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0

    slideList = trainSlideData.getSlideList()
    labelList = trainSlideData.getLabelList()
    for i in range(len(slideList)):
        slide = slideList[i]
        label = [labelList[i]]
        num_label = LE.transform(np.asarray(label))
        if num_label == 0:
            num_0 += len(slide)
        elif num_label == 1:
            num_1 += len(slide)
        elif num_label == 2:
            num_2 += len(slide)
        elif num_label == 3:
            num_3 += len(slide)
        elif num_label == 4:
            num_4 += len(slide)
        elif num_label == 5:
            num_5 += len(slide)

    instances_per_class = [num_0,
                           num_1,
                           num_2,
                           num_3,
                           num_4,
                           num_5]

    print("Real class distribution:\n %s" % instances_per_class)
