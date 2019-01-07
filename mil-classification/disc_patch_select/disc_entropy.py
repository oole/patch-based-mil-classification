import gc
import re

import numpy as np
from scipy.ndimage import gaussian_filter

import data_tf
import predict

from disc_patch_select.disc_finder import AbstractDiscFinder

class OriginalDiscFinder( AbstractDiscFinder ):

    def find_discriminative_patches(self):
        disc_patches = sum(np.count_nonzero(h) for h in self.H)

        return self.H, disc_patches  # , train_predict_accuracy, train_max_accuracy, train_logreg_acccuracy

    def useDuringPredict(self):
        # default is not to use during prediciton, only during training
        # Default: False
        return False