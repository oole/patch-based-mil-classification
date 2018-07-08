import numpy as np
import re
from scipy.ndimage.filters import gaussian_filter
import predict
import netutil

import tensorflow as tf
import dataset
import train
import data_tf
import gc

from sklearn.metrics import accuracy_score, confusion_matrix
import csv
import train_logreg
import util
import evaluate

shuffle_buffer_size = 2048

def emtrain(trainSlideData, valSlideData,
            loadpath, savepath, batch_size,
            initial_epochnum=0,
            model_name='model',
            spatial_smoothing=False,
            do_augment=True,
            num_epochs =2,
            dropout_ratio=0.5,
            learning_rate=0.0005, sanity_check=False, logfile_path=None, logreg_savepath=None, runName="", netAcc=None):

    iteration = 0

    epochnum = initial_epochnum
    # get_per_class_instances(trainSlideData, label_encoder)

    splitTrainSlideData, splitValSlideData = data_tf.splitSlideLists(trainSlideData, valSlideData)

    old_disc_patches = splitTrainSlideData.getNumberOfPatches()
    with tf.Session() as sess:
        ## create iterator
        create_iterator_patch = dataset.slidelist_to_patchlist(splitTrainSlideData.getSlideList())
        create_iterator_dataset = dataset.img_dataset_augment(create_iterator_patch, batch_size=batch_size,
                                                   shuffle_buffer_size=shuffle_buffer_size, shuffle=False,  getlabel=data_tf.getlabel)
        create_iterator_iter = create_iterator_dataset.make_one_shot_iterator()
        if (netAcc is None):
            proxy_iterator_handle_ph= tf.placeholder(tf.string, shape=[])
            proxy_iterator = tf.data.Iterator.from_string_handle(proxy_iterator_handle_ph, output_types=create_iterator_iter.output_types,
                                                             output_shapes=create_iterator_iter.output_shapes)
            x, y = proxy_iterator.get_next()
            netAcc = netutil.build_model(model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)
            netAcc.setIteratorHandle(proxy_iterator_handle_ph)






        # model saver
        saver = tf.train.Saver()
        ########################
        # load model from disc
        saver.restore(sess, loadpath)

        netAcc.getSummmaryWriter(runName, sess.graph)

        while True:

            ##
            H, disc_patches_new, train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy = \
                find_discriminative_patches(splitTrainSlideData,
                                            netAcc,
                                            spatial_smoothing,
                                            sess,
                                            dropout_ratio=dropout_ratio, sanity_check=sanity_check, do_augment=do_augment, logreg_model_savepath=logreg_savepath, epochnum=epochnum)
            print("Discriminative patches: " + repr(disc_patches_new) + ". Before: " +  repr(old_disc_patches))

            gc.collect()

            train_accuracy, val_accuracy = train_on_discriminative_patches(splitTrainSlideData, valSlideData, netAcc, H, initial_epochnum, num_epochs, disc_patches_new,
                                            dropout_ratio=dropout_ratio, learning_rate=learning_rate, sess=sess, do_augment=do_augment, runName=runName)

            epochnum += num_epochs
            old_disc_patches = disc_patches_new
            iteration = iteration + 1

            util.write_log_file(logfile_path, epochNum=epochnum, trainPredictAccuracy=train_predict_accuracy, trainMaxAccuracy=train_max_accuracy,
                                trainLogRegAcccuracy=train_logreg_acccuracy, trainAccuracy=train_accuracy,
                                valAccuracy=val_accuracy)
            if savepath is not None:
                saver.save(sess, savepath)

            print("Iteration done (breaking)")
            # Since there is a memory leak problem at the moment the learning will not be looped
            break


def find_discriminative_patches(trainSlideData, netAccess, spatial_smoothing,
                                sess, dropout_ratio=0.5,sanity_check=False, do_augment=False, logreg_model_savepath=None, epochnum=None):
    slideList = trainSlideData.getSlideList()
    slideLabelList = trainSlideData.getSlideLabelList()
    labelEncoder = trainSlideData.getLabelEncoder()

    batchSize = netAccess.getBatchSize()

    H = initialize_h(slideList)

    # S image level argamx for true class
    S = []

    # Ex dataset level argmax for each class respectively
    E0 = []
    E1 = []
    E2 = []
    E3 = []
    E4 = []
    E5 = []

    # Check patches and their argmax
    # M STEP
    if sanity_check:
        train_histograms = []
        num_0 = 0
        num_1 = 0
        num_2 = 0
        num_3 = 0
        num_4 = 0
        num_5 = 0

    number_of_correct_pred = 0
    train_histograms = []
    for i in range(trainSlideData.getNumberOfSlides()):
        # This is where the spatial structure should be recovered
        # We need to keep a list of discrimative or not
        print("------------------------------------------------------------------------")
        print("Slide:  " + repr(i) + "/" + repr(trainSlideData.getNumberOfSlides() - 1))
        print("Predicting.")
        print("------------------------------------------------------------------------")
        patches = slideList[i]
        # get true label
        label = [slideLabelList[i]]

        if label != np.asarray(data_tf.getlabel(patches[1])):
            raise Exception("ERROR, labels do not correspond")

        num_label = labelEncoder.transform(np.asarray(label))


        pred_iterator_len = len(patches)

        pred_iterator =  trainSlideData.getIterators(netAccess)[i]

        pred_iterator_handle = sess.run(pred_iterator.string_handle())

        slide_y_pred, slide_y_pred_prob, slide_y_pred_argmax = predict.predict_given_net(pred_iterator_handle, pred_iterator_len, netAccess,
                                                                      batch_size=batchSize, dropout_ratio=dropout_ratio, sess=sess)

        pred_histogram = predict.histogram_for_predictions(slide_y_pred_argmax)
        train_histograms.append(pred_histogram)
        if len(slide_y_pred) != len(patches) or sum(pred_histogram) != len(patches):
            Exception("Predictions not corresponding to number of patches!")


        number_of_correct_pred += pred_histogram[num_label]

        if sanity_check:
            """ Check whether computation makes sense"""
            train_histograms.append(pred_histogram)

            print("True label: " + data_tf.getlabel(patches[1]))
            print("Label encoding:" + str(labelEncoder.classes_))
            print(pred_histogram)
            """ end of info"""
        true_label_pred_values = np.asarray(slide_y_pred_prob)[:,0]

        ## Spatial smoothing
        if (spatial_smoothing):
            patch_coordinates = get_patch_coordinates(patches)
            spatial_pred = get_spatial_predictions(trainSlideData.getSlideDimensionList()[i], patches, true_label_pred_values)
            smooth_spatial_pred = gaussian_filter(spatial_pred, sigma=0.5)
            smoothed_predictions = get_patch_predictions_from_probability_map(smooth_spatial_pred, patch_coordinates)
            S.append(smoothed_predictions)
        else:
            S.append(true_label_pred_values)

        if num_label == 0:
            if sanity_check:
                num_0 += len(S[i])
            E0.extend(S[i])
        elif num_label == 1:
            if sanity_check:
                num_1 += len(S[i])
            E1.extend(S[i])
        elif num_label == 2:
            if sanity_check:
                num_2 += len(S[i])
            E2.extend(S[i])
        elif num_label == 3:
            if sanity_check:
                num_3 += len(S[i])
            E3.extend(S[i])
        elif num_label == 4:
            if sanity_check:
                num_4 += len(S[i])
            E4.extend(S[i])
        elif num_label == 5:
            if sanity_check:
                num_5 += len(S[i])
            E5.extend(S[i])
        else:
            Exception("probabilities could not be attributed")

    train_predict_accuracy = number_of_correct_pred / trainSlideData.getNumberOfPatches()
    print("Number of correct predictions: %s, corresponds to %0.3f" % (
    str(number_of_correct_pred), train_predict_accuracy))
    if sanity_check:
        print("Evaluating on training set")
        # at this point it would be cool to evaluate the whole train set, the accuracy and loss should correspond to the
        # trainin accuracy
        # total_eval = cnn_pred.evaluate_generator(data.patchgen_no_shuffle(train_slidelist, batch_size, label_encoder),
        #                                          steps=train_total_number_of_patches // batch_size)
        # print("Evaluation Accuracy: %s" % total_eval[1])

    ### Testing MAX predict ###
    predictions = list(map(labelEncoder.inverse_transform, list(map(np.argmax, train_histograms))))
    train_max_accuracy = accuracy_score(slideLabelList, predictions)
    # print(accuracy)
    confusion = confusion_matrix(slideLabelList, predictions)
    print("Max Accuracy: %0.5f" % train_max_accuracy)
    print("Max Confusion: \n%s" % str(confusion))

    logreg_model = train_logreg.train_logreg_from_histograms_and_labels(train_histograms, slideLabelList)
    train_logreg_acccuracy, train_logreg_confusion = train_logreg.test_given_logreg(train_histograms, slideLabelList, logreg_model)
    print("LogReg Accuracy: %0.5f" % train_logreg_acccuracy)
    print("LogReg Confusion: \n%s" % str(train_logreg_confusion))

    if logreg_model_savepath is not None:
        train_logreg.save_logreg_model(logreg_model, logreg_model_savepath + "_" + str(epochnum) + ".model")


    ### End Testing MAX predict ###
    overall_histo = [sum(np.asarray(train_histograms)[:, 0]),
                     sum(np.asarray(train_histograms)[:, 1]),
                     sum(np.asarray(train_histograms)[:, 2]),
                     sum(np.asarray(train_histograms)[:, 3]),
                     sum(np.asarray(train_histograms)[:, 4]),
                     sum(np.asarray(train_histograms)[:, 5])]

    print("Overall histogram: %s" % overall_histo)

    # """ Max Train """
    # max_accuracy, max_confusion = predict.test_max_predict(train_histograms, label_encoder, slide_label_list)
    #
    # print("Max accuracy: %0.5d\n Max confusion: %s" % (max_accuracy, str(max_confusion)))

    # Remove non discriminative patches
    # E STEP1
    for i in range(trainSlideData.getNumberOfSlides()):
        print("------------------------------------------------------------------------")
        print("Slide" + repr(i) + "/" + repr(trainSlideData.getNumberOfSlides() - 1))
        print("Find discriminizing.")
        print("------------------------------------------------------------------------")
        label = [slideLabelList[i]]
        num_label = labelEncoder.transform(np.asarray(label))

        # in paper:
        #   H_perc = P1 = 5%
        #   P2 = Ex_perc =  30%
        H_perc = 5
        E0_perc = 30  # 'AA'
        E1_perc = 30  # 'AO'
        E2_perc = 30  # 'DA'
        E3_perc = 30  # 'OA'
        E4_perc = 30  # 'OD'
        E5_perc = 30  # '

        Hi = np.percentile(S[i], H_perc)

        if num_label == 0:
            Ri = np.percentile(E0, E0_perc)
        elif num_label == 1:
            Ri = np.percentile(E1, E1_perc)
        elif num_label == 2:
            Ri = np.percentile(E2, E2_perc)
        elif num_label == 3:
            Ri = np.percentile(E3, E3_perc)
        elif num_label == 4:
            Ri = np.percentile(E4, E4_perc)
        elif num_label == 5:
            Ri = np.percentile(E5, E5_perc)

        T = min([Hi, Ri])
        print("Hi= %0.5f, Ri= %0.5f, T= %0.5f" % (Hi, Ri, T))
        patches = slideList[i]
        positive_patches = 0
        for j in range(len(patches)):
            if S[i][j] < T:
                H[i][j] = 0
            else:
                H[i][j] = 1
                positive_patches += 1
        print("Positive patches: %0.f" % (positive_patches))
        print("Total patches: %0.f" % (len(patches)))
        print("Percentage of positive patches: %.2f" % (positive_patches / len(patches)))

    disc_patches = sum(np.count_nonzero(h) for h in H)


    return H, disc_patches, train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy

def train_on_discriminative_patches(trainSlideData, valSlideData, netAccess, H, initial_epochnum, num_epochs, num_patches,
                                    dropout_ratio, learning_rate, sess, do_augment=False, runName=""):

    slideList = trainSlideData.getSlideList()

    batchSize = netAccess.getBatchSize()

    train_patches = dataset.slidelist_to_patchlist(slideList, H=H)

    if len(train_patches) != num_patches:
        raise Exception("H did not work correctly")

    if do_augment:
        train_dataset = dataset.img_dataset_augment(train_patches, batch_size=batchSize,
                                            shuffle_buffer_size=shuffle_buffer_size, shuffle=True, getlabel=data_tf.getlabel_new)
    else:
        train_dataset = dataset.img_dataset(train_patches, batch_size=batchSize,
                                                    shuffle_buffer_size=shuffle_buffer_size, shuffle=True)
    train_iterator = train_dataset.make_initializable_iterator()
    actualEpoch = initial_epochnum
    for i in range(num_epochs):

        train_accuracy = train.train_given_net(netAccess,
                            num_patches, train_iterator,
                            num_epochs=num_epochs, dropout_ratio=dropout_ratio, learning_rate=learning_rate, sess=sess,
                              val_iterator=train_iterator, val_iterator_len=num_patches, runName=runName)


        actualEpoch +=1

        evaluate.evaluateNet(netAccess, None, valSlideData, actualEpoch, sess=sess, dropout=dropout_ratio,
                             runName=runName)
    return train_accuracy


def initialize_h(slidelist):
    H = []
    for slide in slidelist:
        H.append(np.ones(len(slide)))
    return H

def get_spatial_predictions(dim, patches, predictions):
    spatial_pred = np.zeros(dim)
    if (len(patches) == len(predictions)):
        for i in range(len(patches)):
            (x,y) = get_coordinates(patches[i])
            spatial_pred[x][y] = predictions[i]
    else:
        raise Exception('spatial_pred could not be created')
    return spatial_pred

def get_coordinates(patchpath):
    x_group = re.search('patch_x-[0-9]+', patchpath).group(0)
    x = int(x_group[ re.search('x', x_group).start()+2 : ])
    y_group =  re.search('patch_x-[0-9]+_y-[0-9]+', patchpath).group(0)
    y = int(y_group[ re.search('y',y_group).start()+2 : ])
    return (x,y)

def get_patch_coordinates(patches):
    coordinates = []
    for patch in patches:
        coordinates.append(get_coordinates(patch))
    return coordinates

def get_patch_predictions_from_probability_map(prob_map, coords):
    probabilities = []
    for (x,y) in coords:
        probabilities.append(prob_map[x,y])
    return probabilities

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

