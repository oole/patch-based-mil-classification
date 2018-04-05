import tensorflow as tf
import util
from collections import Counter
import numpy as np


def predict_given_net(iterator_handle,
                      pred_iterator_handle, pred_iterator_len,
                      y_pred_op, y_argmax_op,
                      keep_prob_ph, is_training_ph,
                      batch_size=64, dropout_ratio=0.5, sess=tf.Session):

    # sess.run(pred_iterator.initializer)
    print("Prediction:")
    # Do non verbose_validation
    batch_pred_y_pred = []
    batch_pred_y_argmax = []
    i=1
    while True:
        try:
            y_pred, y_argmax = sess.run([y_pred_op, y_argmax_op],
                                        feed_dict={keep_prob_ph: (1 - dropout_ratio),
                                                   is_training_ph: False,
                                                   iterator_handle: pred_iterator_handle})
            util.update_print(
                "Prediction: batch %0.d / %0.d" %
                (i, pred_iterator_len // batch_size + 1))
            batch_pred_y_pred.extend(y_pred)
            batch_pred_y_argmax.extend(y_argmax)
        except tf.errors.OutOfRangeError:
            print("End of prediction dataset.")
            break
        i += 1
    return batch_pred_y_pred, batch_pred_y_argmax


def histogram_for_predictions(predictions):
    counter = Counter(predictions)
    zero = get_count(counter, 0)
    one = get_count(counter, 1)
    two = get_count(counter, 2)
    three = get_count(counter, 3)
    four = get_count(counter, 4)
    five = get_count(counter, 5)
    histogram = np.asarray([zero, one, two, three, four, five])
    return histogram


def get_count(counter, number):
    num = counter.get(number)
    if num is None:
        num = 0
    return num
