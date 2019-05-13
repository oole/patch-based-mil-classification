import gc
import dataset

import numpy as np
import scipy.stats as stats

import data_tf
import predict

from disc_patch_select.disc_original import OriginalDiscFinder

"""
Filters discriminative patches during patch classification aggregation based on the per-class probability entropy.
"""


class EctropyDiscFinder(OriginalDiscFinder):
    """
    Initialize with the given threshold percentile, that is applied to the per-class probability entropy.
    """

    def __init__(self, thresholdPercentile):
        if thresholdPercentile > 100 or thresholdPercentile < 0:
            raise ValueError("Invalid percentile, must be < 100 and >0")
        self.thresholdPercentile = thresholdPercentile

    def find_discriminative_patches(self):
        # disc_patches = sum(np.count_nonzero(h) for h in self.H)
        slideLabels = self.trainSlideData.getSlideLabelList()
        numericLabels = self.trainSlideData.getLabelEncoder().transform(slideLabels)

        split_collected_y_pred_prob, split_collected_y_pred_argmax = self.collectProbabilities()

        totalNumberOfPatches = 0
        for i in range(split_collected_y_pred_argmax.shape[0]):
            self.H[i], numberOfPatches = self.filterDiscriminativePatches(split_collected_y_pred_prob[i],
                                                                          split_collected_y_pred_argmax[i],
                                                                          numericLabels[i])
            totalNumberOfPatches += numberOfPatches

        return self.H, totalNumberOfPatches  # , train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy

    '''
    Collects the probabilities for the given slides
    '''

    def collectProbabilities(self):
        # get information for data
        slideList = self.trainSlideData.getSlideList()
        slideLabelList = self.trainSlideData.getSlideLabelList()
        labelEncoder = self.trainSlideData.getLabelEncoder()
        batchSize = self.netAccess.getBatchSize()

        patchesPerSlide = len(slideList[0])

        # if all the slides have the same number of patches, then it can all be handled by one iterator.
        predIterator, predIteratorInitOp, iteratorLength = \
            self.trainSlideData.getCollectiveIterator(netAccess=self.netAccess, augment=False)

        self.sess.run(predIteratorInitOp)

        pred_iterator_handle = self.sess.run(predIterator.string_handle())

        _, collected_y_pred_prob, collected_y_pred_argmax = predict.predict_given_net(
            pred_iterator_handle,
            iteratorLength,
            self.netAccess,
            batch_size=batchSize,
            dropout_ratio=self.dropout_ratio,
            sess=self.sess)

        split_collected_y_pred_prob = np.reshape(collected_y_pred_prob,
                                                 (self.trainSlideData.getNumberOfSlides(), patchesPerSlide, 3))
        split_collected_y_pred_argmax = np.reshape(collected_y_pred_argmax,
                                                   (self.trainSlideData.getNumberOfSlides(), patchesPerSlide))

        return split_collected_y_pred_prob, split_collected_y_pred_argmax

    def useDuringPredict(self):
        # default is not to use during prediciton, only during training
        # Default: False
        return True

    """
    Filters the given probabilities for patches according to the true class and and the per class probabilites
    Return the Hidden Variable matrix.
    """

    def filterDiscriminativePatches(self, perClassProbabilities, predictedArgmax, trueClassIndex=None):
        numberOfInstances = perClassProbabilities.shape[0]
        H = np.ones(numberOfInstances)

        entropies = []
        for i in range(numberOfInstances):
            entropies.append(stats.entropy(perClassProbabilities[i]))

        # percentile H_perc -> 5%
        H_perc = self.thresholdPercentile * 10

        T = np.percentile(entropies, H_perc)
        positiveInstances = 0
        for i in range(numberOfInstances):
            if (entropies[i] < T):
                H[i] = 1
                positiveInstances += 1
            else:
                H[i] = 0
        print("Entropy: %s , Positive Instances %s/%s" % (
        str(self.thresholdPercentile), str(positiveInstances), str(numberOfInstances)))

        return H, positiveInstances
