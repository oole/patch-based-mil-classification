import tensorflow as tf
import util
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import netutil

def predict_given_net(pred_iterator_handle, pred_iterator_len,
                      netAcc,
                      batch_size=64, dropout_ratio=0.5, sess=tf.Session):

    # sess.run(pred_iterator.initializer)
    print("Prediction:")
    # Do non verbose_validation
    batch_pred_y_pred = []
    batch_pred_y_argmax = []
    batch_pred_y_pred_prob = []
    i=1

    while True:
        try:
            y_pred, y_pred_prob, y_argmax = sess.run([netAcc.getYPred(), netAcc.getYPredProb(), netAcc.getYArgmax()],
                                        feed_dict={netAcc.getKeepProb(): (1 - dropout_ratio),
                                                   netAcc.getIsTraining(): False,
                                                   netAcc.getIteratorHandle: pred_iterator_handle})
            util.update_print(
                "Prediction: batch %0.d / %0.d" %
                (i, pred_iterator_len // batch_size + 1))
            batch_pred_y_pred.extend(y_pred)
            batch_pred_y_argmax.extend(y_argmax)
            batch_pred_y_pred_prob.extend(y_pred_prob)
        except tf.errors.OutOfRangeError:
            print("End of prediction dataset.")
            break
        i += 1
    return batch_pred_y_pred, batch_pred_y_pred_prob, batch_pred_y_argmax


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

def predict_vote(pred_list, label_encoder):
    histogram = histogram_for_predictions(pred_list)
    prediction =  label_encoder.inverse_transform(np.argmax(histogram))
    return prediction

