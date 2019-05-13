import csv
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
import dataset

"""
Check patch name for classname and return it
"""


def getlabel(patch):
    label = ""
    if "OA" in patch:
        label = "OA"
    elif "DA" in patch:
        label = "DA"
    elif "AA" in patch:
        label = "AA"
    elif "OD" in patch:
        label = "OD"
    elif "AO" in patch:
        label = "AO"
    elif "GBM" in patch:
        label = "GBM"
    return label


"""
Check patch name for classname and return it
"""


def getlabel_new(patch):
    label = ""
    if "class-OA" in patch:
        label = "OA"
    elif "class-DA" in patch:
        label = "DA"
    elif "class-AA" in patch:
        label = "AA"
    elif "class-OD" in patch:
        label = "OD"
    elif "class-AO" in patch:
        label = "AO"
    elif "class-GBM" in patch:
        label = "GBM"
    else:
        raise ValueError("label could not be found")
    return label


"""
Returns the labelencoder for LGG, GBM
"""


def labelencoder():
    lE = LabelEncoder()
    lE.fit(["OA", "DA", "AA", "OD", "AO", "GBM"])
    return lE


"""
Load patches from given directory
/dir/patient/patchxy.jpg
"""


def loadpatches(datapath):
    patients = os.listdir(datapath)

    slidelist = []
    slide_dimensions = []
    disc_patches = 0
    # read all patients
    # EM, learning patient wise
    for i in range(len(patients)):
        patientpath = datapath + "/" + patients[i]
        patientdir = patientpath
        slides = os.listdir(patientdir)
        for slide in slides:
            patch_paths = []
            if "TCGA" in slide:
                slidepath = patientpath + "/" + slide
                patches = os.listdir(slidepath)
                if len(patches) < 3:
                    continue
                for patch in patches:
                    if "jpg" in patch:
                        patch_paths.append(slidepath + "/" + patch)
            slide_info = open(slidepath + "/info.txt").read().split("\n")
            slide_dimensions.append((int(slide_info[0][4:]), int(slide_info[1][4:])))
            slidelist.append(patch_paths)
            disc_patches = disc_patches + len(patch_paths)
    patches = []
    for slide in slidelist:
        patches.extend(slide)
    return patches


"""
Load patches from given directory
/dir/patient/patchxy.jpg
"""


def load_patch_folder(datapath):
    patches = os.listdir(datapath)

    regex = re.compile('.*tif')
    ok_patch = [datapath + "/" + x for x in patches if regex.match(x)]

    return ok_patch


"""
Reads slide locations and their respecitve labels from a csv file
"""


def loaddata(dataPath):
    data = csv.reader(open(dataPath))
    slides, labels = [], []
    for row in data:
        labels.append(row[0])
        slides.append(row[1])
    return slides, labels


def split_patches(patches):
    twenty_perc = len(patches) // 5

    x_val = patches[-twenty_perc:]
    x_train = patches[:-twenty_perc]
    return x_train, x_val


def write_histograms_and_labels_to_file(histograms, labels, filepath):
    histopath = filepath + '/histograms.csv'
    labelpath = filepath + '/lists.csv'

    with open(histopath, 'wb') as f:
        pickle.dump(histograms, f)

    with open(labelpath, 'wb') as f:
        pickle.dump(labels, f)


def read_histograms_and_labels_from_file(filepath):
    histopath = filepath + '/histograms.csv'
    labelpath = filepath + '/lists.csv'

    with open(histopath, 'rb') as f:
        histograms = pickle.load(f)

    with open(labelpath, 'rb') as f:
        labels = pickle.load(f)

    return histograms, labels


# def imread(path):
#     img = cv2.imread(path)
#     return img

def float_and_norm(img):
    modimg = img.astype('float32')
    modimg = modimg / 255
    return modimg


def collect_data(datapath, batch_size, filter_batch_size=True):
    patients = os.listdir(datapath)
    np.random.shuffle(patients)
    slidelist = []
    slide_dimensions = []
    patchnumber = 0
    # read all patients
    # EM, learning patient wise
    for i in range(len(patients)):
        print("patient " + str(i))
        patientpath = datapath + "/" + patients[i]
        patientdir = patientpath
        slides = os.listdir(patientdir)
        print("slidelen: " + str(len(slides)))
        for slide in slides:
            patch_paths = []
            if "TCGA" in slide or "slide" in slide:
                slidepath = patientpath + "/" + slide
                patches = os.listdir(slidepath)

                if filter_batch_size and len(patches) - 1 < batch_size:
                    print("SKIP")
                    continue
                for patch in patches:
                    if "tif" in patch:
                        patch_paths.append(slidepath + "/" + patch)
            if filter_batch_size and len(patch_paths) < batch_size:
                continue
            try:
                slide_info = open(slidepath + "/info.txt").read().split("\n")
                slide_dimensions.append((int(slide_info[0][4:]), int(slide_info[1][4:])))
            except:
                # ignore
                print("no txt")
            slidelist.append(patch_paths)
            patchnumber = patchnumber + len(patch_paths)
    # at this point we have a list of lists containing the patches

    slide_label = []
    for slide in slidelist:
        slide_label.append(getlabel(slide[0]))

    if len(slide_label) != len(slidelist):
        raise Exception("Error during data preparation.")

    return slidelist, slide_dimensions, patchnumber, slide_label


def loaddata_csv(dataPath, slide_index=0, class_index=1):
    data = csv.reader(open(dataPath))
    slidepaths, labels, dimensions = [], [], []
    for row in data:
        slide = row[slide_index]
        label = row[class_index]
        slidepaths.append(slide)
        labels.append(label)
        slide_info = open(slide + "/info.txt").read().split("\n")
        dimensions.append((int(slide_info[0][4:]), int(slide_info[1][4:])))
    return slidepaths, labels, dimensions


def collect_flat_data(datapath):
    patch_paths = []
    patches = os.listdir(datapath)
    for patch in patches:
        if "tif" in patch:
            patch_paths.append(datapath + "/" + patch)
    # at this point we have a list of lists containing the patches
    return patch_paths


def collect_data_csv(train_csv, getLabel, doAugment=True):
    slidepaths, labels, dimensions = loaddata_csv(train_csv)

    number_of_patches = 0
    slidelist = []
    for slide in slidepaths:
        patchlist = []
        patches = os.listdir(slide)
        for patch in patches:
            if "patch" in patch:
                patchpath = slide + "/" + patch
                if "jpg" in patch:
                    patchlist.append(patchpath)
        number_of_patches += len(patchlist)
        slidelist.append(patchlist)

    return SlideData(slidelist, dimensions, number_of_patches, labels, getLabel, doAugment)


class SlideData:
    def __init__(self, slideList, slideDimensionList, numberOfPatches, slideLabelList, getLabel, doAugment,
                 labelencoder=labelencoder(), parseFunctionAugment=dataset.input_parser_imglabel_augment,
                 parseFunction=dataset.input_parser_imglabel_no_augment):
        self.slideList = slideList
        self.slideDimensionList = slideDimensionList
        self.numberOfPatches = numberOfPatches
        self.slideLabelList = slideLabelList
        self.getLabel = getLabel
        self.doAugment = doAugment
        self.slideIteratorAugment = None
        self.iteratorInitOpsAugment = None
        self.slideIteratorNormal = None
        self.iteratorInitOpsNormal = None
        self.labelEncoder = labelencoder
        self.parseFunctionAugment = parseFunctionAugment
        self.parseFunction = parseFunction
        self.collectiveIterator = None
        self.collectiveIteratorInitOp = None
        self.collectiveIteratorAugment = None
        self.collectiveIteratorInitOpAugment = None

    def getSlideList(self):
        return self.slideList

    def getSlideDimensionList(self):
        return self.slideDimensionList

    def getNumberOfPatches(self):
        return self.numberOfPatches

    def getSlideLabelList(self):
        return self.slideLabelList

    def getNumberOfSlides(self):
        return len(self.slideList)

    def setLabelEncoder(self, labelEncoder):
        self.labelEncoder = labelEncoder

    def getLabelEncoder(self):
        return self.labelEncoder

    def getLabelFunc(self):
        return self.getLabel

    def getDoAugment(self):
        return self.doAugment

    def getparseFunctionAugment(self):
        return self.parseFunctionAugment

    def getparseFunctionNormal(self):
        return self.parseFunction

    def getIterator(self, netAccess=None, batchSize=None, augment=True):
        if netAccess is not None and batchSize is not None:
            print("Explicit batchSize will overwrite netAccess batchSize")
        if netAccess is not None and batchSize is None:
            batchSize = netAccess.getBatchSize()
        if netAccess is None and batchSize is None:
            raise ValueError("Either netAcess or batchSize must be given.")

        returnIterator = None
        returnIteratorInitOps = None
        # Create if they do not exist yet
        if augment:
            if (self.slideIteratorAugment is None or self.iteratorInitOpsAugment is None):
                print("INFO: Creating datasets and iterators.")
                if self.doAugment:
                    slideDataset = dataset.img_dataset_augment(self.getSlideList()[0], batch_size=batchSize,
                                                               shuffle_buffer_size=None, shuffle=False,
                                                               getlabel=self.getLabelFunc(),
                                                               labelEncoder=self.labelEncoder,
                                                               parseFunctionAugment=self.parseFunctionAugment)

                    iterator = tf.data.Iterator.from_structure(slideDataset.output_types,
                                                               slideDataset.output_shapes)

                    iteratorOps = []
                    for slide in self.getSlideList():
                        slideDataset = dataset.img_dataset_augment(slide, batch_size=batchSize,
                                                                   shuffle_buffer_size=None, shuffle=False,
                                                                   getlabel=self.getLabelFunc(),
                                                                   labelEncoder=self.labelEncoder,
                                                                   parseFunctionAugment=self.parseFunctionAugment)
                        init_op = iterator.make_initializer(slideDataset)
                        iteratorOps.append(init_op)
                    self.slideIteratorAugment = iterator
                    self.iteratorInitOpsAugment = iteratorOps
                else:
                    slideDataset = dataset.img_dataset(self.getSlideList()[0], batch_size=batchSize,
                                                       shuffle_buffer_size=None, shuffle=False,
                                                       getlabel=self.getLabelFunc(),
                                                       labelEncoder=self.labelEncoder,
                                                       parseFunction=self.parseFunction)

                    iterator = tf.data.Iterator.from_structure(slideDataset.output_types,
                                                               slideDataset.output_shapes)

                    iteratorOps = []
                    for slide in self.getSlideList():
                        slideDataset = dataset.img_dataset(slide, batch_size=batchSize,
                                                           shuffle_buffer_size=None, shuffle=False,
                                                           getlabel=self.getLabelFunc(),
                                                           labelEncoder=self.labelEncoder,
                                                           parseFunction=self.parseFunction)
                        init_op = iterator.make_initializer(slideDataset)
                        iteratorOps.append(init_op)
                    self.slideIteratorAugment = iterator
                    self.iteratorInitOpsAugment = iteratorOps

            returnIterator = self.slideIteratorAugment
            returnIteratorInitOps = self.iteratorInitOpsAugment

        else:
            if (self.slideIteratorNormal is None or self.iteratorInitOpsNormal is None):
                print("INFO: Creating datasets and iterators.")
                if self.doAugment:
                    slideDataset = dataset.img_dataset_augment(self.getSlideList()[0], batch_size=batchSize,
                                                               shuffle_buffer_size=None, shuffle=False,
                                                               getlabel=self.getLabelFunc(),
                                                               labelEncoder=self.labelEncoder,
                                                               parseFunctionAugment=self.parseFunction)

                    iterator = tf.data.Iterator.from_structure(slideDataset.output_types,
                                                               slideDataset.output_shapes)

                    iteratorOps = []
                    for slide in self.getSlideList():
                        slideDataset = dataset.img_dataset_augment(slide, batch_size=batchSize,
                                                                   shuffle_buffer_size=None, shuffle=False,
                                                                   getlabel=self.getLabelFunc(),
                                                                   labelEncoder=self.labelEncoder,
                                                                   parseFunctionAugment=self.parseFunction)
                        init_op = iterator.make_initializer(slideDataset)
                        iteratorOps.append(init_op)
                    self.slideIteratorNormal = iterator
                    self.iteratorInitOpsNormal = iteratorOps
                else:
                    slideDataset = dataset.img_dataset(self.getSlideList()[0], batch_size=batchSize,
                                                       shuffle_buffer_size=None, shuffle=False,
                                                       getlabel=self.getLabelFunc(),
                                                       labelEncoder=self.labelEncoder,
                                                       parseFunction=self.parseFunction)

                    iterator = tf.data.Iterator.from_structure(slideDataset.output_types,
                                                               slideDataset.output_shapes)

                    iteratorOps = []
                    for slide in self.getSlideList():
                        slideDataset = dataset.img_dataset(slide, batch_size=batchSize,
                                                           shuffle_buffer_size=None, shuffle=False,
                                                           getlabel=self.getLabelFunc(),
                                                           labelEncoder=self.labelEncoder,
                                                           parseFunction=self.parseFunction)
                        init_op = iterator.make_initializer(slideDataset)
                        iteratorOps.append(init_op)
                    self.slideIteratorNormal = iterator
                    self.iteratorInitOpsNormal = iteratorOps

            returnIterator = self.slideIteratorNormal
            returnIteratorInitOps = self.iteratorInitOpsNormal

        # Return existing iterator and initOps
        return returnIterator, returnIteratorInitOps

    def getCollectiveIterator(self, netAccess=None, batchSize=None, augment=True):
        if netAccess is not None and batchSize is not None:
            print("Explicit batchSize will overwrite netAccess batchSize")
        if netAccess is not None and batchSize is None:
            batchSize = netAccess.getBatchSize()
        if netAccess is None and batchSize is None:
            raise ValueError("Either netAcess or batchSize must be given.")

        returnIterator = None
        returnIteratorInitOp = None
        # Create if they do not exist yet

        collectedPatches = dataset.slidelist_to_patchlist(self.getSlideList())

        iteratorLength = len(collectedPatches)

        if augment:
            if (self.collectiveIteratorAugment is None or self.collectiveIteratorInitOpAugment is None):
                print("INFO: Creating datasets and iterators.")
                if self.doAugment:
                    collectedPatchDataset = dataset.img_dataset_augment(collectedPatches, batch_size=batchSize,
                                                                        shuffle_buffer_size=None, shuffle=False,
                                                                        getlabel=self.getLabelFunc(),
                                                                        labelEncoder=self.labelEncoder,
                                                                        parseFunctionAugment=self.parseFunctionAugment)

                    iterator = tf.data.Iterator.from_structure(collectedPatchDataset.output_types,
                                                               collectedPatchDataset.output_shapes)

                    iteratorOp = iterator.make_initializer(collectedPatchDataset)

                    self.collectiveIteratorAugment = iterator
                    self.collectiveIteratorInitOpAugment = iteratorOp
                else:
                    collectedPatchDataset = dataset.img_dataset(collectedPatches, batch_size=batchSize,
                                                                shuffle_buffer_size=None, shuffle=False,
                                                                getlabel=self.getLabelFunc(),
                                                                labelEncoder=self.labelEncoder,
                                                                parseFunction=self.parseFunction)

                    iterator = tf.data.Iterator.from_structure(collectedPatchDataset.output_types,
                                                               collectedPatchDataset.output_shapes)

                    iteratorOp = iterator.make_initializer(collectedPatchDataset)

                    self.collectiveIteratorAugment = iterator
                    self.collectiveIteratorInitOpAugment = iteratorOp

            returnIterator = self.collectiveIteratorAugment
            returnIteratorInitOp = self.collectiveIteratorInitOpAugment

        # self.collectiveIterator = None
        # self.collectiveIteratorInitOp = None
        else:
            if (self.collectiveIterator is None or self.collectiveIteratorInitOp is None):
                print("INFO: Creating datasets and iterators.")
                if self.doAugment:
                    collectedPatchDataset = dataset.img_dataset_augment(collectedPatches, batch_size=batchSize,
                                                                        shuffle_buffer_size=None, shuffle=False,
                                                                        getlabel=self.getLabelFunc(),
                                                                        labelEncoder=self.labelEncoder,
                                                                        parseFunctionAugment=self.parseFunction)

                    iterator = tf.data.Iterator.from_structure(collectedPatchDataset.output_types,
                                                               collectedPatchDataset.output_shapes)

                    iteratorOp = iterator.make_initializer(collectedPatchDataset)

                    self.collectiveIterator = iterator
                    self.collectiveIteratorInitOp = iteratorOp
                else:
                    collectedPatchDataset = dataset.img_dataset(collectedPatches, batch_size=batchSize,
                                                                shuffle_buffer_size=None, shuffle=False,
                                                                getlabel=self.getLabelFunc(),
                                                                labelEncoder=self.labelEncoder,
                                                                parseFunction=self.parseFunction)

                    iterator = tf.data.Iterator.from_structure(collectedPatchDataset.output_types,
                                                               collectedPatchDataset.output_shapes)

                    iteratorOp = iterator.make_initializer(collectedPatchDataset)

                    self.collectiveIterator = iterator
                    self.collectiveIteratorInitOp = iteratorOp

            returnIterator = self.collectiveIterator
            returnIteratorInitOp = self.collectiveIteratorInitOp

        # Return existing iterator and initOps
        return returnIterator, returnIteratorInitOp, iteratorLength


def splitSlideLists(trainSlideData, valSlideData, splitSeed=None):
    hasDim = True
    if trainSlideData.getSlideDimensionList() is None:
        hasDim = False
        splitResult = train_test_split(trainSlideData.getSlideList(), valSlideData.getSlideList(),
                                       trainSlideData.getSlideLabelList(), valSlideData.getSlideLabelList(),
                                       stratify=trainSlideData.getSlideLabelList(), random_state=splitSeed)
    else:
        splitResult = train_test_split(trainSlideData.getSlideList(), valSlideData.getSlideList(),
                                       trainSlideData.getSlideDimensionList(), valSlideData.getSlideDimensionList(),
                                       trainSlideData.getSlideLabelList(), valSlideData.getSlideLabelList(),
                                       stratify=trainSlideData.getSlideLabelList(), random_state=splitSeed)

    if hasDim:
        trainSlideList = splitResult[0]
        valSlideList = splitResult[3]
        trainDimList = splitResult[4]
        valDimList = splitResult[7]
        trainLabelList = splitResult[8]
        valLabelList = splitResult[11]
    else:
        trainSlideList = splitResult[0]
        valSlideList = splitResult[3]
        trainLabelList = splitResult[4]
        valLabelList = splitResult[7]

    if hasDim:
        if (len(trainSlideList) != len(trainDimList) != len(trainLabelList)):
            raise ValueError("Split is wrong")
        if (len(valSlideList) != len(valDimList) != len(valLabelList)):
            raise ValueError("Split is wrong")
    else:
        if (len(trainSlideList) != len(trainLabelList)):
            raise ValueError("Split is wrong")
        if (len(valSlideList) != len(valLabelList)):
            raise ValueError("Split is wrong")
    if not hasDim:
        trainDimList = None
        valDimList = None

    newTrainSlideData = SlideData(trainSlideList, trainDimList, np.asarray(trainSlideList).size, trainLabelList,
                                  trainSlideData.getLabelFunc(),
                                  trainSlideData.getDoAugment(),
                                  labelencoder=trainSlideData.getLabelEncoder(),
                                  parseFunctionAugment=trainSlideData.getparseFunctionAugment(),
                                  parseFunction=trainSlideData.getparseFunctionNormal())
    newValSlideData = SlideData(valSlideList, valDimList, np.asarray(valSlideList).size, valLabelList,
                                valSlideData.getLabelFunc(),
                                valSlideData.getDoAugment(),
                                labelencoder=valSlideData.getLabelEncoder(),
                                parseFunctionAugment=valSlideData.getparseFunctionAugment(),
                                parseFunction=valSlideData.getparseFunctionNormal())

    return newTrainSlideData, newValSlideData


def KFoldSlideList(trainSlideData, valSlideData, numberOfSplits=5, splitSeed=1337, shuffleSeed=10):
    kf = KFold(n_splits=numberOfSplits, random_state=splitSeed)

    np.random.seed(shuffleSeed)
    idxs = np.arange(len(trainSlideData.getSlideList()))
    np.random.shuffle(idxs)
    for train_idx, test_idx in kf.split(idxs):
        if trainSlideData.getSlideDimensionList() is None:
            hasDim = False
            trainDimList = None
            valDimList = None
        else:
            hasDim = True
            trainDimList = [np.asarray(trainSlideData.getSlideDimensionList()[i]) for i in idxs[train_idx]]
            valDimList = [valSlideData.getSlideDimensionList()[i] for i in idxs[test_idx]]

        trainSlideList = [np.asarray(trainSlideData.getSlideList()[i]) for i in idxs[train_idx]]
        trainLabelList = [np.asarray(trainSlideData.getSlideLabelList()[i]) for i in idxs[train_idx]]
        newTrainSlideData = SlideData(trainSlideList,
                                      None,
                                      np.sum(trainSlideList[i].size for i in range(len(trainSlideList))),
                                      trainLabelList,
                                      trainSlideData.getLabelFunc(),
                                      trainSlideData.getDoAugment(),
                                      labelencoder=trainSlideData.getLabelEncoder(),
                                      parseFunctionAugment=trainSlideData.getparseFunctionAugment(),
                                      parseFunction=trainSlideData.getparseFunctionNormal())
        valSlideList = [np.asarray(valSlideData.getSlideList()[i]) for i in idxs[test_idx]]
        valLabelList = [np.asarray(valSlideData.getSlideLabelList()[i]) for i in idxs[test_idx]]
        newValSlideData = SlideData(trainSlideList,
                                      None,
                                      np.sum(valSlideList[i].size for i in range(len(valSlideList))),
                                      valLabelList,
                                      valSlideData.getLabelFunc(),
                                      valSlideData.getDoAugment(),
                                      labelencoder=valSlideData.getLabelEncoder(),
                                      parseFunctionAugment=valSlideData.getparseFunctionAugment(),
                                      parseFunction=valSlideData.getparseFunctionNormal())
        if hasDim:
            if (len(trainSlideList) != len(trainDimList) != len(trainLabelList)):
                raise ValueError("Split is wrong")
            if (len(valSlideList) != len(valDimList) != len(valLabelList)):
                raise ValueError("Split is wrong")
        else:
            if (len(trainSlideList) != len(trainLabelList)):
                raise ValueError("Split is wrong")
            if (len(valSlideList) != len(valLabelList)):
                raise ValueError("Split is wrong")
        yield newTrainSlideData, newValSlideData






# returns 10
def getTestSizeData(trainSlideData, valSlideData, size):
    newTrainSlideData = SlideData(getSubList(trainSlideData.getSlideList(), size),
                                  getSubList(trainSlideData.getSlideDimensionList(), size),
                                  size, getSubList(trainSlideData.getSlideLabelList(), size),
                                  trainSlideData.getLabelFunc(), trainSlideData.getDoAugment())
    newValSlideData = SlideData(getSubList(valSlideData.getSlideList(), size),
                                getSubList(valSlideData.getSlideDimensionList(), size),
                                size, getSubList(valSlideData.getSlideLabelList(), size), valSlideData.getLabelFunc(),
                                valSlideData.getDoAugment())
    newTrainSlideData.setLabelEncoder(trainSlideData.getLabelEncoder())
    newValSlideData.setLabelEncoder(valSlideData.getLabelEncoder())
    return newTrainSlideData, newValSlideData


def getSubList(list, size):
    return list[:size]
