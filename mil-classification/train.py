import dataset
import tensorflow as tf
import numpy as np
import util

def train_net(train_patchlist, validation_patchlist, num_epochs=2, batch_size=64, do_augment=False,
              dropout_ratio=0.5, lr=0.0005, savepath=None):

    if do_augment:
        # TODO img_dataset_augment
        train_dataset = dataset.img_dataset(train_patchlist, batch_size=batch_size,
                                            shuffle_buffer_size=len(train_patchlist), shuffle=True)
        train_iterator = train_dataset.make_initializable_iterator()
    else:
        train_dataset = dataset.img_dataset(train_patchlist, batch_size=batch_size,
                                            shuffle_buffer_size=len(train_patchlist), shuffle=True)
        train_iterator = train_dataset.make_initializable_iterator()

    val_dataset = dataset.img_dataset(validation_patchlist, batch_size,
                                      shuffle_buffer_size=len(validation_patchlist), shuffle=False)
    val_iterator = val_dataset.make_initializable_iterator()

    ### Fix BN update problem
    extr_up_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ##########

    with tf.Session() as sess:
        iterator_handle, iterator_access, proxy_iterator = dataset.proxy_iterator(sess, train_iterator, val_iterator)

        train_iterator_handle = iterator_access[0]
        val_iterator_handle = iterator_access[1]

        x, y = proxy_iterator.get_next()

        train, loss, y, accuracy, x, keep_prob, learning_rate, is_training = netutil.build_model(
            'hemodel', x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)

        ### SAVER ###
        saver = tf.train.Saver()
        ########################
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            sess.run(train_iterator.initializer)
            sess.run(val_iterator.initializer)
            print("Epoch: %s" % epoch)
            print("Training:")
            batch_train_acc = []
            batch_train_err = []
            i = 1
            while True:
                try:

                    _, err, acc, _ = sess.run([train, loss, accuracy, extr_up_op],
                                              feed_dict={keep_prob: (1 - dropout_ratio),
                                                         learning_rate: lr,
                                                         is_training: True,
                                                         iterator_handle: train_iterator_handle})

                    util.update_print(
                        "Training, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" % (
                        epoch, err, acc, i, len(train_patchlist) // batch_size + 1))
                    i = i + 1
                    batch_train_acc.append(acc)
                    batch_train_err.append(err)
                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break
            print("Epoch %0.d - Training Summary -- Loss: %0.5f, Acc: %0.5f" %
                  (epoch, sum(np.asarray(batch_train_err)) / len(batch_train_err),
                   sum(np.asarray(batch_train_acc)) / len(batch_train_acc)))
            i = 1

            ### Do non verbose_validation
            batch_val_err = []
            batch_val_acc = []
            while True:
                try:
                    err, acc = sess.run([loss, accuracy],
                                        feed_dict={keep_prob: (1 - dropout_ratio),
                                                   is_training: False,
                                                   iterator_handle: val_iterator_handle})
                    i = i + 1
                    batch_val_acc.append(acc)
                    batch_val_err.append(err)
                except tf.errors.OutOfRangeError:
                    print("End of validation dataset.")
                    break
            print("Epoch %0.d - Validation Summary -- Loss: %0.5f, Acc: %0.5f" %
                  (epoch, sum(np.asarray(batch_val_err)) / len(batch_val_err),
                   sum(np.asarray(batch_val_acc)) / len(batch_val_acc)))
        if not savepath==None:
            saver.save(sess, savepath)
