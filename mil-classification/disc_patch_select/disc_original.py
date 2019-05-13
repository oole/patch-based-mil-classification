import gc
import re

import numpy as np
from scipy.ndimage import gaussian_filter

import data_tf
import predict

from disc_patch_select.disc_finder import AbstractDiscFinder

"""
Discriminative patch finder, based on per-patch class probabilities. 

Following Le Hou et al. "Patch-based Convolutional Neural Network for Whole Slide Tissue Image Classification"

"""


class OriginalDiscFinder(AbstractDiscFinder):
    """
    Filters discriminative patches by applying a per image threshold on the true-class probability.
    """

    def find_discriminative_patches(self):
        slideList = self.trainSlideData.getSlideList()
        slideLabelList = self.trainSlideData.getSlideLabelList()
        labelEncoder = self.trainSlideData.getLabelEncoder()

        batchSize = self.netAccess.getBatchSize()

        H = self.initialize_h(slideList)

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
        if self.sanity_check:
            train_histograms = []
            num_0 = 0
            num_1 = 0
            num_2 = 0
            num_3 = 0
            num_4 = 0
            num_5 = 0

        number_of_correct_pred = 0
        train_histograms = []
        predIterator, predIteratorInitOps = self.trainSlideData.getIterator(netAccess=self.netAccess, augment=False)
        for i in range(self.trainSlideData.getNumberOfSlides()):
            gc.collect()
            # This is where the spatial structure should be recovered
            # We need to keep a list of discrimative or not
            print("------------------------------------------------------------------------")
            print("Slide:  " + repr(i) + "/" + repr(self.trainSlideData.getNumberOfSlides() - 1))
            print("Predicting.")
            print("------------------------------------------------------------------------")
            patches = slideList[i]
            # get true label
            label = [slideLabelList[i]]

            if label != np.asarray(self.trainSlideData.getLabelFunc()(patches[1])):
                raise Exception("ERROR, labels do not correspond")

            num_label = labelEncoder.transform(np.asarray(label))

            pred_iterator_len = len(patches)

            self.sess.run(predIteratorInitOps[i])

            pred_iterator_handle = self.sess.run(predIterator.string_handle())

            slide_y_pred, slide_y_pred_prob, slide_y_pred_argmax = predict.predict_given_net(pred_iterator_handle,
                                                                                             pred_iterator_len,
                                                                                             self.netAccess,
                                                                                             batch_size=batchSize,
                                                                                             dropout_ratio=self.dropout_ratio,
                                                                                             sess=self.sess)

            pred_histogram = predict.histogram_for_predictions(slide_y_pred_argmax)
            train_histograms.append(pred_histogram)
            if len(slide_y_pred) != len(patches) or sum(pred_histogram) != len(patches):
                Exception("Predictions not corresponding to number of patches!")

            number_of_correct_pred += pred_histogram[num_label]

            if self.sanity_check:
                """ Check whether computation makes sense"""
                train_histograms.append(pred_histogram)

                print("True label: " + data_tf.getlabel(patches[1]))
                print("Label encoding:" + str(labelEncoder.classes_))
                print(pred_histogram)
                """ end of info"""
            true_label_pred_values = np.asarray(slide_y_pred_prob)[:, 0]

            ## Spatial smoothing
            if (self.spatial_smoothing):
                patch_coordinates = self.get_patch_coordinates(patches)
                spatial_pred = self.get_spatial_predictions(self.trainSlideData.getSlideDimensionList()[i], patches,
                                                            true_label_pred_values)
                smooth_spatial_pred = self.gaussian_filter(spatial_pred, sigma=0.5)
                smoothed_predictions = self.get_patch_predictions_from_probability_map(smooth_spatial_pred,
                                                                                       patch_coordinates)
                S.append(smoothed_predictions)
            else:
                S.append(true_label_pred_values)

            if num_label == 0:
                if self.sanity_check:
                    num_0 += len(S[i])
                E0.extend(S[i])
            elif num_label == 1:
                if self.sanity_check:
                    num_1 += len(S[i])
                E1.extend(S[i])
            elif num_label == 2:
                if self.sanity_check:
                    num_2 += len(S[i])
                E2.extend(S[i])
            elif num_label == 3:
                if self.sanity_check:
                    num_3 += len(S[i])
                E3.extend(S[i])
            elif num_label == 4:
                if self.sanity_check:
                    num_4 += len(S[i])
                E4.extend(S[i])
            elif num_label == 5:
                if self.sanity_check:
                    num_5 += len(S[i])
                E5.extend(S[i])
            else:
                Exception("probabilities could not be attributed")

        overall_histo = [sum(np.asarray(train_histograms)[:, 0]),
                         sum(np.asarray(train_histograms)[:, 1]),
                         sum(np.asarray(train_histograms)[:, 2]),
                         sum(np.asarray(train_histograms)[:, 3]),
                         sum(np.asarray(train_histograms)[:, 4]),
                         sum(np.asarray(train_histograms)[:, 5])]

        print("Overall histogram: %s" % overall_histo)

        # Remove non discriminative patches
        # E STEP1
        for i in range(self.trainSlideData.getNumberOfSlides()):
            print("------------------------------------------------------------------------")
            print("Slide" + repr(i) + "/" + repr(self.trainSlideData.getNumberOfSlides() - 1))
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

        self.H = H
        return H, disc_patches

    def get_patch_predictions_from_probability_map(prob_map, coords):
        probabilities = []
        for (x, y) in coords:
            probabilities.append(prob_map[x, y])
        return probabilities

    def get_spatial_predictions(self, dim, patches, predictions):
        spatial_pred = np.zeros(dim)
        if (len(patches) == len(predictions)):
            for i in range(len(patches)):
                (x, y) = self.get_coordinates(patches[i])
                spatial_pred[x][y] = predictions[i]
        else:
            raise Exception('spatial_pred could not be created')
        return spatial_pred

    def get_coordinates(self, patchpath):
        x_group = re.search('patch_x-[0-9]+', patchpath).group(0)
        x = int(x_group[re.search('x', x_group).start() + 2:])
        y_group = re.search('patch_x-[0-9]+_y-[0-9]+', patchpath).group(0)
        y = int(y_group[re.search('y', y_group).start() + 2:])
        return (x, y)

    def get_patch_coordinates(self, patches):
        coordinates = []
        for patch in patches:
            coordinates.append(self.get_coordinates(patch))
        return coordinates
