import dataset
import tensorflow as tf
import numpy as np
import util
import netutil
import data_tf

def train_net(train_patchlist , validation_patchlist=None, getlabel_train=data_tf.getlabel_new, getlabel_val=data_tf.getlabel, num_epochs=2, batch_size=64, do_augment=True,
              dropout_ratio=0.5, lr=0.0005, savepath=None, shuffle_buffer_size=1984, loadpath=None, model_name="modelname", sess=tf.Session(), log_savepath=None):

    if do_augment:
        # TODO img_dataset_augment
        train_dataset = dataset.img_dataset_augment(train_patchlist, batch_size=batch_size,
                                                    shuffle_buffer_size=shuffle_buffer_size, shuffle=True, getlabel = getlabel_train)
        train_iterator = train_dataset.make_initializable_iterator()
    else:
        train_dataset = dataset.img_dataset(train_patchlist, batch_size=batch_size,
                                                    shuffle_buffer_size=shuffle_buffer_size, shuffle=True, getlabel = getlabel_train)
        train_iterator = train_dataset.make_initializable_iterator()

    val_dataset = dataset.img_dataset(validation_patchlist, batch_size,
                                      shuffle_buffer_size=len(validation_patchlist), shuffle=False, getlabel=getlabel_val)
    val_iterator = val_dataset.make_initializable_iterator()

    iterator_handle, iterator_access, proxy_iterator = dataset.proxy_iterator(sess, train_iterator, val_iterator)

    train_iterator_handle = iterator_access[0]
    val_iterator_handle = iterator_access[1]

    x, y = proxy_iterator.get_next()

    train, loss, y, accuracy, x, keep_prob, learning_rate, is_training, y_pred, y_argmax, y_pred_prob = netutil.build_model(
        model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)

    # Fix BN update problem
    extra_up_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ##########

    # SAVER ###
    saver = tf.train.Saver()
    ########################
    if loadpath is None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, loadpath)

    for epoch in range(num_epochs):
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        print("Epoch: %s/%s" % (epoch+1, num_epochs))
        print("Training:")
        batch_train_acc = []
        batch_train_err = []
        i = 1
        while True:
            try:

                _, err, acc, _, = sess.run([train, loss, accuracy, extra_up_op],
                                           feed_dict={keep_prob: (1 - dropout_ratio),
                                                      learning_rate: lr,
                                                      is_training: True,
                                                      iterator_handle: train_iterator_handle})

                util.update_print(
                    "Training, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" %
                    (epoch+1, err, acc, i, len(train_patchlist) // batch_size + 1))
                i = i + 1
                batch_train_acc.append(acc)
                batch_train_err.append(err)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        print("Epoch %0.d - Training Summary -- Loss: %0.5f, Acc: %0.5f" %
              (epoch+1, sum(np.asarray(batch_train_err)) / len(batch_train_err),
               sum(np.asarray(batch_train_acc)) / len(batch_train_acc)))
        i = 1

        # Do non verbose_validation
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
              (epoch+1, sum(np.asarray(batch_val_err)) / len(batch_val_err),
               sum(np.asarray(batch_val_acc)) / len(batch_val_acc)))
        util.write_log_file(log_savepath, epochnum=epoch, train_accuracy=(sum(np.asarray(batch_train_acc)) / len(batch_train_acc)), val_accuracy=(sum(np.asarray(batch_val_acc)) / len(batch_val_acc)))
    if savepath is not None:
        saver.save(sess, savepath)


def train_given_net(iterator_handle_ph, train_iterator_handle,
                    train_iterator_len, train_iterator,
                    train_op, loss_op, accuracy_op, keep_prob_ph, learning_rate_ph, is_training,
                    val_iterator_handle=None, val_iterator_len=None, val_iterator=None,
                    num_epochs=2, batch_size=64, dropout_ratio=0.5, learning_rate=0.0005, sess=tf.Session()):

    # Fix BN update problem
    extra_up_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ##########

    for epoch in range(num_epochs):
        sess.run(train_iterator.initializer)

        print("Epoch: %s/%s" % (epoch+1, num_epochs))
        print("Training:")
        batch_train_acc = []
        batch_train_err = []
        i = 1
        while True:
            try:

                _, err, acc, _ = sess.run([train_op, loss_op, accuracy_op, extra_up_op],
                                          feed_dict={keep_prob_ph: (1 - dropout_ratio),
                                                     learning_rate_ph: learning_rate,
                                                     is_training: True,
                                                     iterator_handle_ph: train_iterator_handle})

                util.update_print(
                    "Training, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" %
                    (epoch+1, err, acc, i, train_iterator_len // batch_size + 1))
                i = i + 1
                batch_train_acc.append(acc)
                batch_train_err.append(err)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        train_accuracy = sum(np.asarray(batch_train_acc)) / len(batch_train_acc)

        print("Epoch %0.d - Training Summary -- Loss: %0.5f, Acc: %0.5f" %
              (epoch+1, sum(np.asarray(batch_train_err)) / len(batch_train_err),
               train_accuracy))

        if val_iterator is not None:
            i = 1
            sess.run(val_iterator.initializer)
            # Do non verbose_validation
            batch_val_err = []
            batch_val_acc = []
            while True:
                try:
                    err, acc = sess.run([loss_op, accuracy_op],
                                        feed_dict={keep_prob_ph: (1 - dropout_ratio),
                                                   is_training: False,
                                                   iterator_handle_ph: val_iterator_handle})
                    # util.update_print(
                    #     "Validation, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" %
                    #     (epoch + 1, err, acc, i, val_iterator_len // batch_size + 1))
                    i = i + 1
                    batch_val_acc.append(acc)
                    batch_val_err.append(err)
                except tf.errors.OutOfRangeError:
                    print("End of validation dataset.")
                    break
            val_accuracy = sum(np.asarray(batch_val_acc)) / len(batch_val_acc)
            print("Epoch %0.d - Validation Summary -- Loss: %0.5f, Acc: %0.5f" %
                  (epoch+1, sum(np.asarray(batch_val_err)) / len(batch_val_err),
                   val_accuracy))

    return train_accuracy, val_accuracy