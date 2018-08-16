import dataset
import tensorflow as tf
import numpy as np
import util
import netutil
import data_tf
from sklearn.utils import shuffle
from skimage import io
import validate
import evaluate
import train_logreg


TBOARDFOLDER="/home/oole/tboard/"

EPOCHNUMBER = 1

def train_net(trainSlideData , valSlideData=None, getlabel_train=data_tf.getlabel_new, num_epochs=2, batch_size=64, do_augment=True,
              dropout_ratio=0.5, lr=0.0005, savepath=None, shuffle_buffer_size=2048, loadpath=None, model_name="modelname", sess=tf.Session(), log_savepath=None, runName="",
              buildNet= netutil.build_model):
    trainSlideData, valSlideData = data_tf.splitSlideLists(trainSlideData, valSlideData)
    train_patches = dataset.slidelist_to_patchlist(trainSlideData.getSlideList())
    val_patches = dataset.slidelist_to_patchlist(valSlideData.getSlideList())
    np.random.shuffle(train_patches)
    if do_augment:
        train_dataset = dataset.img_dataset_augment(train_patches, batch_size=batch_size,
                                                    shuffle_buffer_size=shuffle_buffer_size, shuffle=True,
                                                    getlabel = getlabel_train,
                                                    labelEncoder=trainSlideData.getLabelEncoder(),
                                                    parseFunctionAugment=trainSlideData.getparseFunctionAugment())
        train_iterator = train_dataset.make_initializable_iterator()
    else:
        train_dataset = dataset.img_dataset(train_patches, batch_size=batch_size,
                                                    shuffle_buffer_size=shuffle_buffer_size, shuffle=True, getlabel = getlabel_train)
        train_iterator = train_dataset.make_initializable_iterator()

    val_dataset = dataset.img_dataset(val_patches, batch_size=batch_size,
                                      getlabel=getlabel_train,
                                      labelEncoder=valSlideData.getLabelEncoder(),
                                      parseFunction=valSlideData.getparseFunctionNormal())

    val_iterator = val_dataset.make_initializable_iterator()

    iterator_handle, iterator_access, proxy_iterator = dataset.proxy_iterator(sess, train_iterator)

    val_iterator.string_handle()
    train_iterator_handle = iterator_access[0]
    # val_iterator_handle = iterator_access[1]

    x, y = proxy_iterator.get_next()

    # train, loss, y, accuracy, x, keep_prob, learning_rate, is_training, y_pred, y_argmax, y_pred_prob = netutil.build_model(
    #     model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True)

    netAcc = buildNet(
        model_name, x, y, use_bn_1=True, use_bn_2=True, use_dropout_1=True, use_dropout_2=True, batchSize=batch_size)
    netAcc.setIteratorHandle(iterator_handle)

    # SAVER ###
    saver = tf.train.Saver()
    ########################
    if loadpath is None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, loadpath)

    actualEpoch = 0
    for i in range(num_epochs):
        trainAccuracy, valAccuracy = train_given_net(netAcc,
                        len(train_patches), train_iterator,
                        val_iterator_len=len(val_patches), val_iterator=val_iterator,
                        num_epochs=1, batch_size=batch_size, dropout_ratio=dropout_ratio, learning_rate=lr, sess=sess, runName=runName, log_savepath= log_savepath)

        # Train logreg model with current net
        logregModel = train_logreg.train_logreg(netAcc, log_savepath, trainSlideData, dropout_ratio, sess)

        evaluate.evaluateNet(netAcc, logregModel, valSlideData, actualEpoch, sess=sess, dropout=dropout_ratio,
                             runName=runName)
        actualEpoch += 1
        global EPOCHNUMBER
        EPOCHNUMBER = actualEpoch+1

    if savepath is not None:
        saver.save(sess, savepath)
    return trainAccuracy, valAccuracy, netAcc


def train_given_net(netAcc,
                    train_iterator_len, train_iterator,
                    val_iterator_len=None, val_iterator=None,
                    num_epochs=2, batch_size=64, dropout_ratio=0.5, learning_rate=0.0005, sess=tf.Session(), runName="", log_savepath=None, actualEpoch=0):
    train_iterator_handle = sess.run(train_iterator.string_handle())

    for epoch in range(num_epochs):
        sess.run(train_iterator.initializer)
        # sess.run(val_iterator.initializer)
        print("Epoch: %s/%s" % (epoch+1, num_epochs))
        print("Training:")
        batch_train_acc = []
        batch_train_err = []
        i = 1
        while True:
            try:

                # ADD IMAGE TO SUMMARY (Sanity check)
                # _, err, acc, _, step, imgSummary = sess.run([netAcc.getTrain(), netAcc.getLoss(), netAcc.getAccuracy(), netAcc.getUpdateOp(), netAcc.getGlobalStep(), netAcc.getImageSummary()],
                _, acc, err, _, step = sess.run(
                    [netAcc.getTrain(), netAcc.getAccuracy(), netAcc.getLoss(), netAcc.getUpdateOp(),
                     netAcc.getGlobalStep()],
                                           feed_dict={netAcc.getKeepProb(): (1 - dropout_ratio),
                                                      netAcc.getLearningRate(): learning_rate,
                                                      netAcc.getIsTraining(): True,
                                                      netAcc.getIteratorHandle(): train_iterator_handle})

                util.update_print(
                    "Training, Epoch: %0.f -- Loss: %0.5f, Acc: %0.5f, %0.d / %0.d. Step: %s" %
                    (EPOCHNUMBER, err, acc, i, train_iterator_len // batch_size + 1, str(step)))
                i = i + 1
                batch_train_acc.append(acc)
                batch_train_err.append(err)
                util.writeBatchStatsToTensorBoard(err, acc, netAcc.getSummmaryWriter(runName, sess.graph), step)
                # ADD IMAGE TO SUMMARY (Sanity check)
                # netAcc.getSummmaryWriter(runName, sess.graph).add_summary(imgSummar
                # y, global_step=step)
                # summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="acc", simple_value=acc)]),global_step=step)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
        trainLoss = sum(np.asarray(batch_train_err)) / len(batch_train_err)
        trainAccuracy = sum(np.asarray(batch_train_acc)) / len(batch_train_acc)
        util.writeEpochStatsToTensorBoard(trainLoss, trainAccuracy, netAcc.getSummmaryWriter(runName, sess.graph), step)
        print("Epoch %0.d - Training Summary -- Loss: %0.5f, Acc: %0.5f" %
              (EPOCHNUMBER, trainLoss, trainAccuracy))
        i = 1

        # for validation, 1. do overall accuracy (patchbased)
        valLoss, valAccuracy = validate.validate_existing_net(val_iterator, val_iterator_len, netAcc, dropout_ratio=dropout_ratio,
                                       batch_size=batch_size, sess=sess)

        util.writeValStatsToTensorBoard(valLoss, valAccuracy, netAcc.getSummmaryWriter(runName, sess.graph), step)

        # 2. do max acc

        # 3. do logreg acc (logreg model needs to be trained first

        util.write_log_file(log_savepath, epochNum=epoch + 1, trainLoss= trainLoss,
                            trainAccuracy=trainAccuracy, valLoss=valLoss,
                            valAccuracy=valAccuracy)
        actualEpoch += 1

    netAcc.getSummmaryWriter(runName, sess.graph).flush()
    return trainAccuracy, valAccuracy