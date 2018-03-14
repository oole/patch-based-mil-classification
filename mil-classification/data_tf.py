import csv
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import openslide as osl
import pickle
from skimage import color
from skimage.io import imsave, imread, imshow, show
from keras import utils
import re

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
    twenty_perc = len(patches)//5

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
    modimg = modimg/255
    return modimg


def patchgen_no_shuffle(slidelist, batch_size, label_encoder, H=None):
    return patchgenval(slidelist, batch_size, label_encoder, H, shuffle=False)

def patchgenval(slidelist, batch_size, label_encoder, H=None, shuffle=True, get_patch_label=getlabel):
    patches = []
    for i in range(len(slidelist)):
        slide = slidelist[i]
        if (H == None):
            for patch in slide:
                patches.append(patch)
        else:
            h = H[i]
            if (len(h) != len(slide)):
                raise Exception("Hidden vars do not correspond to patches")
            for j in range(len(slide)):
                if (h[j] > 0):
                    patches.append(slide[j])

    if shuffle:
        np.random.shuffle(patches)
    batchimg = []
    batchlabel = []
    while (True):
        for i in range(len(patches)):
            patch = patches[i]
            patchimg = float_and_norm(imread(patch))
            batchimg.append(patchimg)
            batchlabel.append(get_patch_label(patch))
            if (i + 1) % batch_size == 0:
                # print("patches yielded: %s" % str(i+1))
                yield (np.asarray(batchimg),
                       utils.to_categorical(label_encoder.transform(np.asarray(batchlabel)), len(label_encoder.classes_)))
                batchimg = []
                batchlabel = []

def evalpatchgen(patches, batchSize, label_encoder, shuffle=True):
    if shuffle:
        np.random.shuffle(patches)
    batchimg = []
    batchlabel = []
    while (True):
        for i in range(len(patches)):
            patch = patches[i]
            patchimg = float_and_norm(imread(patch))
            batchimg.append(patchimg)
            batchlabel.append(getlabel(patch))
            if (i + 1) % batchSize == 0:
                # print("Length: %s, %s"  % (str(len(batchimg)), str(len(utils.to_categorical(label_encoder.transform(np.asarray(batchlabel)), len(label_encoder.classes_))))))
                yield (np.asarray(batchimg),
                       utils.to_categorical(label_encoder.transform(np.asarray(batchlabel)), len(label_encoder.classes_)))
                batchimg = []
                batchlabel = []




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
                #ignore
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


def collect_flat_data(datapath):
    patch_paths = []
    patches = os.listdir(datapath)
    for patch in patches:
        if "tif" in patch:
            patch_paths.append(datapath + "/" + patch)
    # at this point we have a list of lists containing the patches



    return patch_paths