import data
import tensorflow as tf
import dataset
import netutil

""" CONFIGURATION """
dropout_ratio = 0.5
number_of_epochs = 1
batch_size = 32

""" LOAD DATA """
training_data_path = '/home/oole/Documents/UKN/BP/validation'
train_slidelist, train_slide_dimensions, train_num_patches, train_slide_label = data.collect_data(training_data_path,
                                                                                                      batch_size)

patches = dataset.slidelist_to_patchlist(train_slidelist)
train_imgs = tf.constant(patches)

train_labels = dataset.get_labels_for_patches(patches)

tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))

tr_data = tr_data.map(dataset.input_parser_imglabel)
tr_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data)

with tf.Session() as sess:
    sess.run(training_init_op)

    train, loss, y, accuracy, x, keep_prob, learning_rate = netutil.build_model('hemodel')

    sess.run(tf.global_variables_initializer())

    for epoch in range(number_of_epochs):
        while True:
            try:
                elem = sess.run(next_element)

                print(elem)
            except tf.errors.OutOfRangeError:
                print("Endof training dataset.")
                break


