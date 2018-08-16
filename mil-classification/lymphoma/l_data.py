from skimage import io
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from data_tf import SlideData
import tensorflow as tf
from random import randint


# read patches and labels from folder

# label parser
"""
Check patch name for classname and return it
"""
def getlabel(patch):
    label = ""
    if "CLL" in patch:
        label = "CLL"
    elif "FL" in patch:
        label = "FL"
    elif "MCL" in patch:
        label = "MCL"
    return label

"""
Returns the labelencoder for LGG, GBM
"""
def labelencoder():
    lE = LabelEncoder()
    lE.fit(["CLL", "FL", "MCL"])
    return lE

"""
Read the patches and their slides from the given datapath
"""
def collect_data(datapath):
    classes = os.listdir(datapath)
    slidelist = []
    slide_dimensions = []
    patchnumber = 0
    # read all patients
    # EM, learning patient wise
    for i in range(len(classes)):
        print("Loading class %s/%s" % (str(i+1), str(len(classes))))
        patientpath = datapath + classes[i]
        patientdir = patientpath
        slides = os.listdir(patientdir)
        for slide in slides:
            patch_paths = []
            slidepath = patientpath + "/" + slide
            patches = os.listdir(slidepath)
            for patch in patches:
                if "jpg" in patch:
                    patch_paths.append(slidepath + "/" + patch)
            slidelist.append(patch_paths)
            patchnumber = patchnumber + len(patch_paths)
    # at this point we have a list of lists containing the patches

    slide_label = []
    for slide in slidelist:
        slide_label.append(getlabel(slide[0]))

    if len(slide_label) != len(slidelist):
        raise Exception("Error during data preparation. (labels do not correspond to slides")

    slideData = SlideData(slidelist, None, patchnumber, slide_label, getlabel, True,
                          labelencoder=labelencoder(),
                          parseFunctionAugment=l_image_augment,
                          parseFunction=l_image_noaugment)
    return slideData

def l_image_augment(img_path, img_label):
    one_hot = tf.one_hot(img_label, 3)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_float = tf.image.convert_image_dtype(img_decoded, tf.float32)
    img_flip_lr = tf.image.random_flip_left_right(img_float)
    img_flip_ud = tf.image.random_flip_up_down(img_flip_lr)
    # random rotation
    rot = tf.random_uniform([],0, 4, dtype=tf.int32)
    img_rot = tf.image.rot90(img_flip_ud, k = rot)
    # Add more augmentation
    img_contrast = tf.image.random_contrast(img_rot, 0.7,1.3)
    img_brightness = tf.image.random_brightness(img_contrast, 0.3)
    img_hue = tf.image.random_hue(img_brightness, 0.09)
    img_saturation = tf.image.random_saturation(img_hue, 0.7, 1.3)
    # End add more augmentation
    img_crop = tf.image.crop_to_bounding_box(img_saturation, 0,0,64,64)
    img_standard = tf.image.per_image_standardization(img_crop)
    return img_standard, one_hot

def l_image_noaugment(img_path, img_label):
    one_hot = tf.one_hot(img_label, 3)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_float = tf.image.convert_image_dtype(img_decoded, tf.float32)
    # End add more augmentation
    img_crop = tf.image.crop_to_bounding_box(img_float, 0,0,64,64)
    img_standard = tf.image.per_image_standardization(img_crop)
    return img_standard, one_hot