import tensorflow as tf
import numpy as np
import util
import predict
import dataset


def validate_existing_net(iterator_handle, input_iterator, iterator_len,
                          dropout_ratio, batch_size,
                          loss_op, accuracy_op,
                          keep_prob_ph, is_training_ph, iterator_handle_ph, sess):

    i = 1
    sess.run(input_iterator.initializer)
    # Do non verbose_validation
    batch_val_err = []
    batch_val_acc = []
    print("Do validation")
    while True:
        try:
            err, acc = sess.run([loss_op, accuracy_op],
                                feed_dict={keep_prob_ph: (1 - dropout_ratio),
                                           is_training_ph: False,
                                           iterator_handle_ph: iterator_handle})
            util.update_print(
                " Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" %
                (err, acc, i, iterator_len // batch_size + 1))
            i = i + 1
            batch_val_acc.append(acc)
            batch_val_err.append(err)
        except tf.errors.OutOfRangeError:
            print("End of validation dataset.")
            break
    print("Validation Summary -- Loss: %0.5f, Acc: %0.5f" %
          (sum(np.asarray(batch_val_err)) / len(batch_val_err),
           sum(np.asarray(batch_val_acc)) / len(batch_val_acc)))


# For validation the slidelist should be 400x400 patches, so that the evaluation is consistent
def evaluate_max_prediction(iterator_handle, validation_slidelist, validation_labels, netAcc, label_encoder):
    predicted_labels = []
    for patches in validation_slidelist:
        # CREATE ITERATOR
        number_of_patches = len(patches)
        val_dataset = dataset.img_dataset(patches, batch_size=netAcc.getBatchSize(),
                                          shuffle_buffer_size=netAcc.getShuffleBufferSize(), shuffle=False)

        val_iterator = val_dataset.make_initializable_iterator()

        # PREDICT LIST OF PATCHES
        batch_pred_y_pred, batch_pred_y_pred_prob, batch_pred_y_argmax = predict.predict_given_net(iterator_handle=iterator_handle,
                                  pred_iterator_handle=val_iterator, pred_iterator_len=number_of_patches,)

        # MAJORITY VOTE
        predict.predict_vote(batch_pred_y_argmax, label_encoder)
        # ADD TO PREDICTED LABELS


    # calculate accuracy
    return #accuracy and confusion matrix
