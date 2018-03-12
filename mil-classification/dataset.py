import tensorflow as tf
import data

def input_parser(img_path):
    label_encoder = data.labelencoder()
    label =  label_encoder.transform([data.getlabel(img_path)])
    one_hot = tf.one_hot(label, 6)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, one_hot

def input_parser_imglabel(img_path, img_label):
    one_hot = tf.one_hot(img_label, 6)

    print(img_path)
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, one_hot

def slidelist_to_patchlist(slidelist, H = None):
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
    return patches

def get_labels_for_patches(patches):
    labels = []
    labelencoder = data.labelencoder()
    for patch in patches:
        labels.append(data.getlabel(patch))
    encoded_labels = labelencoder.transform(labels)
    return encoded_labels