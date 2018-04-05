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

shuffle_buffer_size = 2000


def emtrain(train_datapath, val_datapath,
            loadpath, savepath,
            label_encoder, batch_size,
            initial_epochnum=0,
            model_name='model',
            spatial_smoothing=False,
            do_augment=True,
            num_epochs =2,
            dropout_ratio=0.5,
            learning_rate=0.0005, sanity_check=False, logfile_path=None, logreg_savepath=None):

    train_slidelist, train_slide_dimensions, total_num_train_patches, train_slide_label = data_tf.collect_data(train_datapath, batch_size)

    val_slidelist, val_slide_dimension, val_slide_num, val_slide_label = data_tf.collect_data(val_datapath,
                                                                                                     batch_size)
    iteration = 0

    epochnum = initial_epochnum
    get_per_class_instances(train_slidelist, train_slide_label, label_encoder)

    old_disc_patches = total_num_train_patches
    with tf.Session() as sess:
        ## create iterator
        val_patches = dataset.slidelist_to_patchlist(val_slidelist)

        val_dataset = dataset.img_dataset(val_patches, batch_size=batch_size,
                                                   shuffle_buffer_size=shuffle_buffer_size, shuffle=False)

        val_iterator = val_dataset.make_one_shot_iterator()

        proxy_iterator_handle_ph= tf.placeholder(tf.string, shape=[])
        proxy_iterator = tf.data.Iterator.from_string_handle(proxy_iterator_handle_ph, output_types=val_iterator.output_types,
                                                       output_shapes=val_iterator.output_shapes)

        x, y = proxy_iterator.get_next()



        train_op, loss_op, y, accuracy_op, x, keep_prob_ph, learning_rate_ph, is_training_ph, y_pred_op, y_argmax_op = \
            netutil.build_model(model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)

        # model saver
        saver = tf.train.Saver()
        ########################
        # load model from disc
        saver.restore(sess, loadpath)

        while True:

            ##
            H, disc_patches_new, train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy = \
                find_discriminative_patches(train_slidelist, train_slide_label,
                                            train_slide_dimensions, total_num_train_patches,
                                            spatial_smoothing,
                                            y_pred_op, y_argmax_op, batch_size, dropout_ratio, label_encoder,
                                            keep_prob_ph, is_training_ph, proxy_iterator_handle_ph,
                                            sess, sanity_check=sanity_check, do_augment=do_augment, logreg_model_savepath=logreg_savepath, epochnum=epochnum)
            print("Discriminative patches: " + repr(disc_patches_new) + ". Before: " +  repr(old_disc_patches))

            gc.collect()

            train_accuracy, val_accuracy = train_on_discriminative_patches(train_slidelist, H, num_epochs, disc_patches_new, proxy_iterator_handle_ph, batch_size,
                                            train_op, loss_op, accuracy_op,
                                            keep_prob_ph, learning_rate_ph, is_training_ph,
                                            dropout_ratio=dropout_ratio, learning_rate=learning_rate, sess=sess, do_augment=do_augment)

            epochnum += num_epochs
            old_disc_patches = disc_patches_new
            iteration = iteration + 1

            util.write_log_file(logfile_path, epochnum=epochnum, train_predict_accuracy=train_predict_accuracy, train_max_accuracy=train_max_accuracy,
                   train_logreg_acccuracy=train_logreg_acccuracy, train_accuracy=train_accuracy,
                   val_accuracy=val_accuracy)
            if savepath is not None:
                saver.save(sess, savepath)

            print("Iteration done (breaking)")
            # Since there is a memory leak problem at the moment the learning will not be looped
            break


def find_discriminative_patches(slide_list, slide_label_list, slide_dimension_list, total_patch_num, spatial_smoothing,
                                y_pred_op, y_argmax_op, batch_size, dropout_ratio, label_encoder,
                                keep_prob_ph, is_training_ph, iterator_handle_ph,
                                sess, sanity_check=False, do_augment=False, logreg_model_savepath=None, epochnum=None):
    H = initialize_h(slide_list)

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
    for i in range(len(slide_list)):
        # This is where the spatial structure should be recovered
        # We need to keep a list of discrimative or not
        print("------------------------------------------------------------------------")
        print("Slide:  " + repr(i) + "/" + repr(len(slide_list) - 1))
        print("Predicting.")
        print("------------------------------------------------------------------------")
        patches = slide_list[i]
        # get true label
        label = [slide_label_list[i]]

        if label != np.asarray(data_tf.getlabel(patches[1])):
            raise Exception("ERROR, labels do not correspond")

        num_label = label_encoder.transform(np.asarray(label))


        pred_iterator_len = len(patches)

        if do_augment:
            pred_dataset = dataset.img_dataset_augment(patches, batch_size=batch_size,
                                               shuffle_buffer_size=shuffle_buffer_size, shuffle=False)
        else:
            pred_dataset = dataset.img_dataset(patches, batch_size=batch_size,
                                                       shuffle_buffer_size=shuffle_buffer_size, shuffle=False)

        pred_iterator = pred_dataset.make_one_shot_iterator()

        pred_iterator_handle = sess.run(pred_iterator.string_handle())

        slide_y_pred, slide_y_pred_argmax = predict.predict_given_net(iterator_handle_ph,
                                                                      pred_iterator_handle, pred_iterator_len,
                                                                      y_pred_op, y_argmax_op,
                                                                      keep_prob_ph, is_training_ph,
                                                                      batch_size=batch_size, dropout_ratio=dropout_ratio, sess=sess)

        pred_histogram = predict.histogram_for_predictions(slide_y_pred_argmax)
        train_histograms.append(pred_histogram)
        if len(slide_y_pred) != len(patches) or sum(pred_histogram) != len(patches):
            Exception("Predictions not corresponding to number of patches!")


        number_of_correct_pred += pred_histogram[num_label]

        if sanity_check:
            """ Check whether computation makes sense"""
            train_histograms.append(pred_histogram)

            print("True label: " + data_tf.getlabel(patches[1]))
            print("Label encoding:" + str(label_encoder.classes_))
            print(pred_histogram)
            """ end of info"""
        true_label_pred_values = np.asarray(slide_y_pred)[:,0]

        ## Spatial smoothing
        if (spatial_smoothing):
            patch_coordinates = get_patch_coordinates(patches)
            spatial_pred = get_spatial_predictions(slide_dimension_list[i], patches, true_label_pred_values)
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

    train_predict_accuracy = number_of_correct_pred / total_patch_num
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
    predictions = list(map(label_encoder.inverse_transform, list(map(np.argmax, train_histograms))))
    train_max_accuracy = accuracy_score(slide_label_list, predictions)
    # print(accuracy)
    confusion = confusion_matrix(slide_label_list, predictions)
    print("Max Accuracy: %0.5f" % train_max_accuracy)
    print("Max Confusion: \n%s" % str(confusion))

    logreg_model = train_logreg.train_logreg_from_histograms_and_labels(train_histograms, slide_label_list)
    train_logreg_acccuracy, train_logreg_confusion = train_logreg.test_given_logreg(train_histograms, slide_label_list, logreg_model)
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
    for i in range(len(slide_list)):
        print("------------------------------------------------------------------------")
        print("Slide" + repr(i) + "/" + repr(len(slide_list) - 1))
        print("Find discriminizing.")
        print("------------------------------------------------------------------------")
        label = [slide_label_list[i]]
        num_label = label_encoder.transform(np.asarray(label))

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
        patches = slide_list[i]
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

def train_on_discriminative_patches(train_slidelist, H, num_epochs, num_patches, iterator_handle_ph, batch_size,
                                    train_op, loss_op, accuracy_op,
                                    keep_prob_ph, learning_rate_ph, is_training_ph,
                                    dropout_ratio, learning_rate, sess, do_augment=False):


    train_patches = dataset.slidelist_to_patchlist(train_slidelist, H=H)

    if len(train_patches) != num_patches:
        raise Exception("H did not work correctly")

    if do_augment:
        train_dataset = dataset.img_dataset_augment(train_patches, batch_size=batch_size,
                                            shuffle_buffer_size=shuffle_buffer_size, shuffle=True)
    else:
        train_dataset = dataset.img_dataset(train_patches, batch_size=batch_size,
                                                    shuffle_buffer_size=shuffle_buffer_size, shuffle=True)
    train_iterator = train_dataset.make_initializable_iterator()

    train_iterator_handle = sess.run(train_iterator.string_handle())

    train_accuracy = train.train_given_net(iterator_handle_ph, train_iterator_handle,
                        num_patches, train_iterator,
                        train_op, loss_op, accuracy_op, keep_prob_ph, learning_rate_ph, is_training_ph,
                        num_epochs=num_epochs, batch_size=batch_size, dropout_ratio=dropout_ratio, learning_rate=learning_rate, sess=sess,
                          val_iterator=train_iterator, val_iterator_handle=train_iterator_handle, val_iterator_len=num_patches)
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



def get_per_class_instances(slide_list, label_list, LE):

    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0

    for i in range(len(slide_list)):
        slide = slide_list[i]
        label = [label_list[i]]
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
