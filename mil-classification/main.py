import data_tf
import tensorflow as tf
import dataset
import netutil
import util

""" CONFIGURATION """
dropout_ratio = 0.5
number_of_epochs = 10
batch_size = 32
lr = 0.0005

""" LOAD DATA """
training_data_path = 'D:/Data/tf_test_data/validation'
train_slidelist, train_slide_dimensions, train_num_patches, train_slide_label = data_tf.collect_data(training_data_path,
                                                                                                      batch_size)

patches = dataset.slidelist_to_patchlist(train_slidelist)
no_patches = len(patches)

train_imgs = tf.constant(patches)

train_labels = dataset.get_labels_for_patches(patches)

tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
tr_data = tr_data.map(dataset.input_parser_imglabel)
tr_data_shuffled = tr_data.shuffle(buffer_size=no_patches)
tr_data_batched = tr_data_shuffled.batch(batch_size)


iterator = tf.data.Iterator.from_structure(tr_data_batched.output_types,tr_data_batched.output_shapes)

x,y = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data_batched)

with tf.Session() as sess:

    train, loss, y, accuracy, x, keep_prob, learning_rate, is_training = netutil.build_model('hemodel', x, y)

    sess.run(tf.global_variables_initializer())

    epoch_train_acc = []
    epoch_train_err = []
    epoch_val_acc = []
    epoch_val_err = []
    for epoch in range(number_of_epochs):
        print("Epoch: %s" % epoch)
        batch_train_acc = []
        batch_train_err = []
        batch_val_err = []
        batch_val_acc = []
        print("Training:")
        i = 1
        while True:
            try:
                sess.run(training_init_op)
                train_x, train_y = sess.run(next_element)

                _, err, acc = sess.run([train, loss, accuracy],
                                       feed_dict={x: train_x,
                                                  y: train_y,
                                                  keep_prob: (1-dropout_ratio),
                                                  learning_rate: lr,
                                                  is_training: True})

                util.update_print("Training, Epoch: %0.d -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" % (epoch, err, acc, i, no_patches))
                i = i+1
                batch_train_acc.append(acc)
                batch_train_err.append(err)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        print("Epoch %0.d summary -- Loss: %0.5d, Acc: %0.5d" % (epoch, sum(batch_train_err)/len(batch_train_err), sum(batch_train_acc)/len(batch_train_acc)))
        print("Validation:")
        while True:
            try:
                sess.run(training_init_op)
                train_x, train_y = sess.run(next_element)

                _, err, acc = sess.run([train, loss, accuracy],
                                       feed_dict={x: train_x,
                                                  y: train_y,
                                                  keep_prob: (1 - dropout_ratio),
                                                  learning_rate: lr,
                                                  is_training: False})

                util.update_print("Validation Epoch: %0.d -- Loss: %0.5f, Acc: %0.5f" % (epoch,err, acc))
                batch_val_err.append(err)
                batch_val_acc.append(acc)
            except tf.errors.OutOfRangeError:
                print("End of validation dataset.")
                break
        epoch_train_acc.append(sum(batch_train_acc)/len(batch_train_acc))
        epoch_train_err.append(sum(batch_train_err)/len(batch_train_err))
        epoch_val_acc.append(sum(batch_val_acc)/len(batch_val_acc))
        epoch_val_err.append(sum(batch_val_err)/len(batch_val_err))

util.plot_train_val_acc(epoch_train_acc, epoch_val_acc)


