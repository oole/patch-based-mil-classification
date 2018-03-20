import data_tf
import tensorflow as tf
import dataset
import netutil
import util
import numpy as np

""" CONFIGURATION """
dropout_ratio = 0.5
number_of_epochs = 5
batch_size = 64
lr = 0.0001

#### log dir
log_directory = "/home/oole/tf_log"

""" LOAD DATA """
# training_data_path = 'D:/Data/tf_test_data/validation'
training_data_path = '/home/oole/tf_test_data/validation'
train_slidelist, train_slide_dimensions, train_num_patches, train_slide_label = data_tf.collect_data(training_data_path,
                                                                                                      batch_size)

patches = dataset.slidelist_to_patchlist(train_slidelist)
no_patches = len(patches)

""" CREATE TRAINING AND VALIDATION DATASET """

train_dataset = dataset.img_dataset(patches, batch_size, shuffle=True, shuffle_buffer_size=no_patches)
train_iterator = train_dataset.make_initializable_iterator()

val_dataset = dataset.img_dataset(patches, batch_size, shuffle=False, shuffle_buffer_size=no_patches)
val_iterator = val_dataset.make_initializable_iterator()


with tf.Session() as sess:
    iterator_handle, iterator_access, proxy_iterator = dataset.proxy_iterator(sess, train_iterator, val_iterator)


    train_iterator_handle = iterator_access[0]
    val_iterator_handle = iterator_access[1]

    x, y = proxy_iterator.get_next()


    train, loss, y, accuracy, x, keep_prob, learning_rate, is_training = netutil.build_model(
        'hemodel', x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)
    ### summary writer ###
    writer = tf.summary.FileWriter(log_directory, graph=tf.get_default_graph())
    extr_up_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ### SAVER ###
    saver = tf.train.Saver()
    saver.restore(sess, '/home/oole/testmodel')
    ########################

    epoch_train_acc = []
    epoch_train_err = []
    epoch_val_acc = []
    epoch_val_err = []
    for epoch in range(number_of_epochs):
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        print("Epoch: %s" % epoch)
        batch_train_acc = []
        batch_train_err = []
        batch_val_err = []
        batch_val_acc = []
        print("Training:")
        i = 1
        while True:
            try:

                _, err, acc,_ = sess.run([train, loss, accuracy, extr_up_op],
                                       feed_dict={keep_prob: (1-dropout_ratio),
                                                  learning_rate: lr,
                                                  is_training: True,
                                                  iterator_handle: train_iterator_handle})

                util.update_print(
                    "Training, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" % (epoch, err, acc, i, no_patches//batch_size+1))
                i = i+1
                batch_train_acc.append(acc)
                batch_train_err.append(err)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        print("Epoch %0.d - Training Summary -- Loss: %0.5f, Acc: %0.5f" %
              (epoch, sum(np.asarray(batch_train_err))/len(batch_train_err),
               sum(np.asarray(batch_train_acc))/len(batch_train_acc)))
        print("Validation:")
        i=1
        while True:
            try:
                err, acc = sess.run([loss, accuracy],
                                       feed_dict={keep_prob: (1-dropout_ratio),
                                                  is_training: False,
                                                  iterator_handle: val_iterator_handle})
                util.update_print(
                    "Validation, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.f / %0.f" % (epoch, err, acc, i, no_patches//batch_size +1))
                i = i + 1
                batch_val_acc.append(acc)
                batch_val_err.append(err)
            except tf.errors.OutOfRangeError:
                print("End of validation dataset.")
                break
        print("Epoch %0.d - Validation Summary -- Loss: %0.5f, Acc: %0.5f" % (
            epoch, sum(np.asarray(batch_val_err)) / len(batch_val_err),
            sum(np.asarray(batch_val_acc)) / len(batch_val_acc)))
        epoch_train_acc.append(sum(batch_train_acc)/len(batch_train_acc))
        epoch_train_err.append(sum(batch_train_err)/len(batch_train_err))
        epoch_val_acc.append(sum(batch_val_acc)/len(batch_val_acc))
        epoch_val_err.append(sum(batch_val_err)/len(batch_val_err))
    save_path = saver.save(sess, '/home/oole/testmodel')
util.plot_train_val_acc(epoch_train_acc, epoch_val_acc, title="")


