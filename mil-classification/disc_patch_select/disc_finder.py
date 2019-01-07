import numpy as np

class AbstractDiscFinder():

    def __init__(self, trainSlideData, netAccess, spatial_smoothing,
                                    sess, dropout_ratio=0.5,sanity_check=False, do_augment=False, logreg_model_savepath=None, epochnum=None):
        self.trainSlideData = trainSlideData
        self.netAccess = netAccess
        self.spatial_smoothing = spatial_smoothing
        self.sess = sess
        self.dropout_ratio = dropout_ratio
        self.sanity_check = sanity_check
        self.do_augment = do_augment
        self.logreg_model_savepath = logreg_model_savepath
        self.epochnum = epochnum
        self.H = self.initialize_h(self.trainSlideData.getSlideList())

        # nothing to do

    def find_discriminative_patches(self):

        disc_patches = sum(np.count_nonzero(h) for h in self.H)

        return self.H, disc_patches #, train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy

    def initialize_h(self, slidelist):
        H = []
        for slide in slidelist:
            H.append(np.ones(len(slide)))
        return H

    def useDuringPredict(self):
        # default is not to use during prediciton, only during training
        # Default: False
        return False