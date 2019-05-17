import validate
import predict
import tensorflow as tf
import data_tf
import netutil
import numpy as np
import train_logreg
import util


from sklearn.metrics import accuracy_score, confusion_matrix

def evaluateNet(netAccess: netutil.NetAccess, logRegModel, valSlideList: data_tf.SlideData, step, sess=tf.Session(), dropout=0.5, runName="", discriminativePatchFinder= None):


    iterator, iteratorInitOps = valSlideList.getIterator(netAccess, augment=False)
    modelNum = 0
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        slideHistograms = []
        simpleAccuracies = []
        discriminativePatchFinder.setThreshold(th)
        for i in range(valSlideList.getNumberOfSlides()):
            iteratorLen = len(valSlideList.getSlideList()[i])
            sess.run(iteratorInitOps[i])
            iteratorHandle = sess.run(iterator.string_handle())
            slide_y_pred, slide_y_pred_prob, slide_y_pred_argmax = \
                predict.predict_given_net(iteratorHandle, iteratorLen,
                                          netAccess, batch_size=netAccess.getBatchSize(), dropout_ratio=dropout, sess=sess,
                                          discriminativePatchFinder=discriminativePatchFinder)
            simpleAccuracy = accuracy_score([valSlideList.getSlideLabelList()[i]] * len(slide_y_pred_argmax), list(
                map(valSlideList.getLabelEncoder().inverse_transform, slide_y_pred_argmax)))
            simpleAccuracies.append(simpleAccuracy)
            histogram = predict.histogram_for_predictions(slide_y_pred_argmax)
            slideHistograms.append(histogram)

        maxPredictions = list(map(valSlideList.getLabelEncoder().inverse_transform, list(map(np.argmax, slideHistograms))))
        maxAccuracy = accuracy_score(valSlideList.getSlideLabelList(), maxPredictions)
        # print(accuracy)
        maxConfusionMatrix = confusion_matrix(valSlideList.getSlideLabelList(), maxPredictions)
        print("Max Confusion Matrix:\n %s" % maxConfusionMatrix)


        if (logRegModel[modelNum] is not None):
            logregAccuracy, logregConfusionMatrix = train_logreg.test_given_logreg(slideHistograms, valSlideList.getSlideLabelList(), logRegModel[modelNum])
            util.writeScalarSummary(logregAccuracy, "logRegAccuracyVal_th-" + str(th), netAccess.getSummmaryWriter(runName, sess.graph), step=step)
            print("LogReg Confusion Matrix:\n %s" % logregConfusionMatrix)

        # scalar, scalarName, summaryWriter, step
        util.writeScalarSummary(sum(simpleAccuracies)/valSlideList.getNumberOfSlides(), "simpleAccuracyVal_th-" + str(th), netAccess.getSummmaryWriter(runName, sess.graph),
                                step=step)
        util.writeScalarSummary(maxAccuracy, "maxAccuracyVal_th-" + str(th), netAccess.getSummmaryWriter(runName, sess.graph), step=step)

        netAccess.getSummmaryWriter(runName, sess.graph).flush()
