import data
import tensorflow as tf
import dataset
import netutil

""" CONFIGURATION """
dropout_ratio = 0.5
number_of_epochs = 10
batch_size = 32
lr = 0.0005

""" LOAD DATA """
training_data_path = '/home/oole/Documents/UKN/BP/validation_jpg'
train_slidelist, train_slide_dimensions, train_num_patches, train_slide_label = data.collect_data(training_data_path,
                                                                                                      batch_size)

patches = dataset.slidelist_to_patchlist(train_slidelist)
train_imgs = tf.constant(patches)

train_labels = dataset.get_labels_for_patches(patches)

tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
# tr_data = tf.data.Dataset.from_tensor_slices((train_imgs))

tr_data = tr_data.map(dataset.input_parser_imglabel)
tr_data_batched = tr_data.batch(batch_size)

iterator = tf.data.Iterator.from_structure(tr_data_batched.output_types,tr_data_batched.output_shapes)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data_batched)

with tf.Session() as sess:

    train, loss, y, accuracy, x, keep_prob, learning_rate, is_training = netutil.build_model('hemodel')

    sess.run(tf.global_variables_initializer())

    for epoch in range(number_of_epochs):
        print("Epoch: %s" % epoch)
        try:
            sess.run(training_init_op)
            print("Training:")
            train_x, train_y = sess.run(next_element)

            _, err, acc = sess.run([train, loss, accuracy],
                                   feed_dict={x: train_x,
                                              y: train_y,
                                              keep_prob: (1-dropout_ratio),
                                              learning_rate: lr,
                                              is_training: True})

            print("Loss: %0.5f, Acc: %0.5f" % (err, acc))
        except tf.errors.OutOfRangeError:
            print("Endof training dataset.")
            break
        try:
            sess.run(training_init_op)
            print("Validation:")
            train_x, train_y = sess.run(next_element)

            _, err, acc = sess.run([train, loss, accuracy],
                                   feed_dict={x: train_x,
                                              y: train_y,
                                              keep_prob: (1 - dropout_ratio),
                                              learning_rate: lr,
                                              is_training: False})

            print("Loss: %0.5f, Acc: %0.5f" % (err, acc))
        except tf.errors.OutOfRangeError:
            print("Endof validation dataset.")
            break



