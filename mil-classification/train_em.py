import os
import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
import data
from datetime import datetime
#import tgm
import re
from scipy.ndimage.filters import gaussian_filter
import csv
import train_log_reg
import predict_cnn
import predict
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, confusion_matrix

tgc = None #tgm.tgClient()

def emtrain(train_datapath, val_datapath,
            loadpath, savepath,
            label_encoder, batch_size,
            initial_epochnum=0,
            model_name='model',
            spatial_smoothing=False,
            do_augment=True,
            tgc_send=False):

    train_slidelist, train_slide_dimensions, old_disc_patches, train_slide_label = data.collect_data(train_datapath, batch_size)

    train_num_patches = old_disc_patches

    val_slidelist, val_slide_dimensions, val_num_patches, val_slide_label = data.collect_data(val_datapath, batch_size)

    train_total_number_of_patches = old_disc_patches
    iteration = 0

    epochnum = initial_epochnum

    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0
    H = initialize_h(train_slidelist)
    get_per_class_instances(train_slidelist, train_slide_label, label_encoder, tgc_send)

    with tf.session()

    while (epochnum < 100):
        if tgc_send:
            tgc.sendmsg("EM Iteration: " + repr(iteration))
        # S image level argamx for true class
        S = []

        # Ex dataset level argmax for each class respectively
        E0 = []
        E1 = []
        E2 = []
        E3 = []
        E4 = []
        E5 = []

        if tgc_send:
            tgc.sendmsg("Predicting patches. (M step)")
        # Check patches and their argmax
        # M STEP
        train_histograms = []
        number_of_correct_preds = 0
        for i in range(len(train_slidelist)):
            # This is where the spatial structure should be recovered
            # We need to keep a list of discrimative or not
            print("------------------------------------------------------------------------")
            print("Slide:  " + repr(i) + "/" + repr(len(train_slidelist)-1))
            print("Predicting.")
            print("------------------------------------------------------------------------")
            patches = train_slidelist[i]
            # get true label
            label = [train_slide_label[i]]
            num_label = label_encoder.transform(np.asarray(label))

            pred, pred_histogram = predict_cnn.predict_one_slide_as_patches(cnn_learn, patches, batch_size, getsub=do_augment)

            number_of_correct_preds += pred_histogram[num_label]

            if len(pred) != len(patches) or sum(pred_histogram) != len(patches):
                Exception("Predictions not corresponding to number of patches!")

            # evaluation = cnn_learn.evaluate_generator(data.evalpatchgen(patches, batchsize, LE),
            #                                     steps = len(patches) //batchsize +1)
            #
            #
            # print("Evaluation. Loss: %0.00f. Accuracy: %0.00f " % (evaluation[0], evaluation[1]))

            train_histograms.append(pred_histogram)
            """ Some more info """
            if label != np.asarray(data.getlabel(patches[1])):
                tgc.sendmsg("ERROR, labels do not correspond")
            print("True label: " + data.getlabel(patches[1]))
            print("Label encoding:" + str(label_encoder.classes_))
            print(pred_histogram)
            """ end of info"""
            corresponding_label_predictions = pred[:, num_label]
            ## Spatial smoothing
            if (spatial_smoothing):
                patch_coordinates = get_patch_coordinates(patches)
                spatial_pred = get_spatial_predictions(train_slide_dimensions[i], patches, corresponding_label_predictions)
                smooth_spatial_pred = gaussian_filter(spatial_pred, sigma=0.5)
                smoothed_predictions = get_patch_predictions_from_probability_map(smooth_spatial_pred, patch_coordinates)
                S.append(smoothed_predictions)
            else:
                S.append(corresponding_label_predictions)

            if num_label == 0:
                num_0 += len(S[i])
                E0.extend(S[i])
            elif num_label == 1:
                num_1 += len(S[i])
                E1.extend(S[i])
            elif num_label == 2:
                num_2 += len(S[i])
                E2.extend(S[i])
            elif num_label == 3:
                num_3 += len(S[i])
                E3.extend(S[i])
            elif num_label == 4:
                num_4 += len(S[i])
                E4.extend(S[i])
            elif num_label == 5:
                num_5 += len(S[i])
                E5.extend(S[i])
            else:
                tgc.sendmsg("probabilities could not be attributed")
                Exception("probabilities could not be attributed")

        print("Number of correct predictions: %s, corresponds to %0.3f" % (str(number_of_correct_preds), number_of_correct_preds/train_num_patches))
        total_eval = cnn_pred.evaluate_generator(data.patchgen_no_shuffle(train_slidelist, batch_size, label_encoder),
                                                 steps=train_total_number_of_patches // batch_size)

        print("Evaluation Accuracy: %s" % total_eval[1])

        overall_histo = [sum(np.asarray(train_histograms)[:,0]),
                         sum(np.asarray(train_histograms)[:,1]),
                         sum(np.asarray(train_histograms)[:,2]),
                         sum(np.asarray(train_histograms)[:,3]),
                         sum(np.asarray(train_histograms)[:,4]),
                         sum(np.asarray(train_histograms)[:,5])]

        if tgc_send:
            tgc.sendmsg("Overall Histogram:\n %s" % overall_histo)
        print("Overall histogram: %s" % overall_histo)
        """ Max Train """
        max_accuracy, max_confusion = predict.test_max_predict(train_histograms, label_encoder, train_slide_label)

        if tgc_send:
            tgc.sendmsg("Accuracy after epoch %0.f: %0.5f\nConfusion:\n%s" % (epochnum, max_accuracy, str(max_confusion)))

        with open(savepath + "_max_accuracy_train.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epochnum, max_accuracy])

        """ Logreg Val """
        train_log_reg.train_logreg_from_histograms_and_labels2(train_histograms, train_slide_label, savepath + "_" + str(epochnum) + '_logreg.model')
        accuracy, confusion = train_log_reg.test_logreg2(train_histograms, train_slide_label, savepath + "_" + str(epochnum) + '_logreg.model')
        if tgc_send:
            tgc.sendmsg("Accuracy after epoch %0.f: %0.5f\nConfusion\n%s" % (epochnum, accuracy, str(confusion)))
        with open(savepath + "_logreg_accuracy_train.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epochnum, accuracy])

        """ Start Model Validation """
        # Predict val
        _, val_histograms = predict_cnn.predict_from_patches(cnn_pred, val_slidelist, batch_size, getsub=False)
        # Max prediction
        val_max_acc, val_max_conf = predict.test_max_predict(val_histograms, label_encoder, val_slide_label)

        # LogReg prediction
        val_logr_acc, val_logr_conf = train_log_reg.test_logreg2(val_histograms, val_slide_label,
                                                         savepath + "_" + str(epochnum) + '_logreg.model')
        #
        if tgc_send:
            tgc.sendmsg("----------------------\n"
                    "Val Accuracy, E: %0.f\n"
                    "Max Acc: %0.5f\nConfusion:\n%s\n"
                    "LogR Acc: %0.5f\nConfusion:\n%s\n"
                    "----------------------" % (epochnum, val_max_acc, val_max_conf,val_logr_acc, val_logr_conf))

        with open(savepath + "_max_accuracy_val.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epochnum, val_max_acc])
        with open(savepath + "_logreg_accuracy_val.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epochnum,val_logr_acc])
        """ End Model Validation """

        if tgc_send:
            tgc.sendmsg("Removing non discriminative patches. (E Step)")
        # Remove non discriminative patches
        # E STEP1
        with open(savepath + "_" + str(epochnum) + '_discrimating.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Label', 'Hi', 'Ri', 'T', 'TotalPatch', 'DisciminativePatch', 'Percentage'])
        for i in range(len(train_slidelist)):
            print("------------------------------------------------------------------------")
            print("Slide" + repr(i) + "/" + repr(len(train_slidelist)-1))
            print("Find discriminizing.")
            print("------------------------------------------------------------------------")
            label = [train_slide_label[i]]
            num_label = label_encoder.transform(np.asarray(label))

            # in paper:
            #   H_perc = P1 = 5%
            #   P2 = Ex_perc =  30%
            H_perc = 5
            E0_perc = 30    # 'AA'
            E1_perc = 30    # 'AO'
            E2_perc = 30    # 'DA'
            E3_perc = 30    # 'OA'
            E4_perc = 30    # 'OD'
            E5_perc = 30    # '

            Hi = np.percentile(S[i], H_perc)

            if num_label == 0:
                Ri = np.percentile(E0,E0_perc)
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
            patches = train_slidelist[i]
            positive_patches = 0
            for j in range(len(patches)):
                if S[i][j] < T:
                    H[i][j] = 0
                else:
                    H[i][j] = 1
                    positive_patches += 1
            print("Positive patches: %0.f" % (positive_patches))
            print("Total patches: %0.f" % (len(patches)))
            print("Percentage of positive patches: %.2f" % (positive_patches/len(patches)))

            with open(savepath + "_" + str(epochnum) + '_discrimating.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epochnum, label[0], Hi, Ri, T, len(patches), positive_patches, positive_patches/len(patches)])

        disc_patches = sum(np.count_nonzero(h) for h in H)

        print("Discriminative patches: "  + repr(disc_patches) + ". Before: " +  repr(old_disc_patches))
        if tgc_send:
            tgc.sendmsg("Patches before: " + repr(old_disc_patches) + "\n" + "Patches now: " + repr(disc_patches))
        # if (old_disc_patches == disc_patches):
        #     keepGoing = False
        #     break

        # Train two epochs
        num_epochs = epochnum + 2
        time_per_batch = 5
        if tgc_send:
            tgc.sendmsg("Training, Iteration: " + repr(iteration))
        print("------------------------------------------------------------------------")
        print("Training patches: " + repr(disc_patches))
        now = datetime.now()
        print('%s/%s/%s - %s:%s:%s' % (repr(now.day), repr(now.month), repr(now.year), repr(now.hour), repr(now.minute), repr(now.second)))
        print("Iteration:" + repr(i))
        print("Trained Epochs:" + repr(epochnum))
        print("Approximate duration: " + repr((disc_patches // batch_size) * time_per_batch))
        print("------------------------------------------------------------------------")
        try:
            if do_augment:
                cnn_learn.fit_generator(data.modpatchgen(train_slidelist, batch_size, label_encoder, H),
                                        steps_per_epoch=disc_patches // batch_size,
                                        epochs=num_epochs, verbose=1,
                                        callbacks=[TensorBoard(log_dir='C:/tflog/'),
                                         ModelCheckpoint(savepath + "_" + repr(epochnum + 2) + "_ckpt.h5"),
                                         CSVLogger(loggerpath, separator=',', append=True), learingrate_callback()],
                                        initial_epoch=epochnum)
            else:
                cnn_learn.fit_generator(data.patchgenval(train_slidelist, batch_size, label_encoder, H),
                                        steps_per_epoch=disc_patches // batch_size,
                                        epochs=num_epochs, verbose=1,
                                        callbacks=[TensorBoard(log_dir='C:/tflog/'),
                                             ModelCheckpoint(savepath + "_" + repr(epochnum + 2) + "_ckpt.h5"),
                                             CSVLogger(loggerpath, separator=',', append=True), learingrate_callback()],
                                        initial_epoch=epochnum)

            # EVALUATION
            cnn_pred.set_weights(cnn_learn.get_weights())
            evaluation = cnn_pred.evaluate_generator(data.patchgenval(val_slidelist, batch_size, label_encoder),
                                                     steps=val_num_patches // batch_size)

            print("Validation Loss: %s, Acc:  %s " % (evaluation[0], evaluation[1]))

            epochnum += 2
            print("Iteration: " + repr(iteration), "Trained epochs: " + repr(epochnum))
        except Exception:
            tgc.sendmsg("Iteration: " + repr(iteration) + "\n" + "Epoch: " + repr(i) + "/")
            raise

        old_disc_patches = disc_patches
        iteration = iteration + 1
        cnn_learn.save(savepath + "_epoch-" + repr(num_epochs) + ".h5")
        if tgc_send:
            tgc.sendmsg("Saved epoch # " + repr(num_epochs))
    cnn_learn.save(savepath + "_epoch-" + repr(num_epochs) + ".h5")

class learingrate_callback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))
        with open(global_savepath + "_learning_rate.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, K.eval(lr), K.eval(lr_with_decay)])


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
        tgc.sendmsg("Cannot create spatial prediction matrix")
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



def get_per_class_instances(slide_list, label_list, LE, tgc_send):

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
    if tgc_send:
        tgc.sendmsg("Real class distribution:\n %s" % instances_per_class)

