#from tensorflow.contrib.data import Dataset, Iterator
import tensorflow as tf
import data

def input_parser(img_path):
    label_encoder = data.labelencoder()
    label =  label_encoder.transform([data.getlabel(img_path)])
    one_hot = tf.one_hot(label, 6)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, one_hot


img_decoded, one_hot = input_parser("GBM")
