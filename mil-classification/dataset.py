import tensorflow as tf
import data_tf

def input_parser(img_path):
    label_encoder = data_tf.labelencoder()
    label =  label_encoder.transform([data_tf.getlabel(str(img_path))])
    one_hot = tf.one_hot(label, 6)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, one_hot

def input_parser_imglabel(img_path, img_label):
    one_hot = tf.one_hot(img_label, 6)

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_decoded = tf.image.convert_image_dtype(img_decoded, tf.float32)
    img_decoded = tf.image.per_image_standardization(img_decoded)


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
    labelencoder = data_tf.labelencoder()
    for patch in patches:
        labels.append(data_tf.getlabel(patch))
    encoded_labels = labelencoder.transform(labels)
    return encoded_labels

def img_dataset(images, batch_size, shuffle_buffer_size=None, shuffle=False):
    labels = get_labels_for_patches(images)
    if shuffle and shuffle_buffer_size == None:
        raise Exception("If shuffle==True shuffle_buffer_size must be set!")

    tr_data = tf.data.Dataset.from_tensor_slices((images, labels))
    tr_data = tr_data.map(input_parser_imglabel)
    if shuffle == True:
        tr_data = tr_data.shuffle(buffer_size=shuffle_buffer_size)
    tr_data = tr_data.batch(batch_size)
    return tr_data

def proxy_iterator(sess, *iterators):
    if len(iterators) < 2:
        raise ValueError("At least two iterators needed in order to chain them")
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, output_types=iterators[0].output_types, output_shapes=iterators[0].output_shapes)
    access = list(map(lambda x: sess.run(x.string_handle()), iterators))
    return handle, access, iterator