import tensorflow as tf
import numpy as np
import util
import predict
import dataset


def validate_existing_net(valIterator, valIteratorLen,  netAcc, dropout_ratio, batch_size, sess):
    i = 1
    valIteratorHandle = sess.run(valIterator.string_handle())
    sess.run(valIterator.initializer)
    # Do non verbose_validation
    batch_val_err = []
    batch_val_acc = []
    print("Do validation")
    while True:
        try:
            err, acc = sess.run([netAcc.getLoss(), netAcc.getAccuracy()],
                                feed_dict={netAcc.getKeepProb(): (1 - dropout_ratio),
                                           netAcc.getIsTraining(): False,
                                           netAcc.getIteratorHandle(): valIteratorHandle})
            util.update_print(
                " Loss: %0.5f, Acc: %0.5f, %0.d / %0.d" %
                (err, acc, i, valIteratorLen // batch_size + 1))
            i = i + 1
            batch_val_acc.append(acc)
            batch_val_err.append(err)
        except tf.errors.OutOfRangeError:
            print("End of validation dataset.")
            break
    valLoss = sum(np.asarray(batch_val_err)) / len(batch_val_err)
    valAccuracy = sum(np.asarray(batch_val_acc)) / len(batch_val_acc)
    print("Validation Summary -- Loss: %0.5f, Acc: %0.5f" %
          (valLoss, valAccuracy))

    return valLoss, valAccuracy


# For validation the slidelist should be 400x400 patches, so that the evaluation is consistent
def evaluate_max_prediction(valSlideList, netAcc):
    predicted_labels = []
    for slide in valSlideList.getSlideList():
        # CREATE ITERATOR
        number_of_patches = len(slide)
        val_dataset = dataset.img_dataset(slide, batch_size=netAcc.getBatchSize(),
                                          shuffle_buffer_size=netAcc.getShuffleBufferSize(), shuffle=False)

        val_iterator = val_dataset.make_initializable_iterator()

        # PREDICT LIST OF PATCHES
        batch_pred_y_pred, batch_pred_y_pred_prob, batch_pred_y_argmax = predict.predict_given_net(iterator_handle=netAcc.getIteratorHandle(),
                                  pred_iterator_handle=val_iterator, pred_iterator_len=number_of_patches,)

        # MAJORITY VOTE
        predict.predict_vote(batch_pred_y_argmax, valSlideList.getLabelEncoder())
        # ADD TO PREDICTED LABELS


    # calculate accuracy
    return #accuracy and confusion matrix
