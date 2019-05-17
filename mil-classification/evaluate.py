import validate
import predict
import tensorflow as tf
import data_tf
import netutil
import numpy as np
import train_logreg
import util


from sklearn.metrics import accuracy_score, confusion_matrix

def evaluateNet(netAccess: netutil.NetAccess, logRegModel, fullLogreg, valSlideData: data_tf.SlideData, step, sess=tf.Session(), dropout=0.5, runName="", discriminativePatchFinder= None):
    iterator, iteratorInitOps = valSlideData.getIterator(netAccess, augment=False)
    slide_raw_probabilities = []
    slide_per_class_probabilities = []
    slide_predicted_argmax = []
    for i in range(valSlideData.getNumberOfSlides()):
        slideIteratorLen = len(valSlideData.getSlideList()[i])
        sess.run(iteratorInitOps[i])
        slideIteratorHandle = sess.run(iterator.string_handle())
        raw_per_class_activations, per_class_probabilities, predicted_argmax = \
            predict.predict_given_net(slideIteratorHandle, slideIteratorLen,
                                      netAccess, batch_size=netAccess.getBatchSize(), dropout_ratio=dropout, sess=sess,
                                      discriminativePatchFinder=discriminativePatchFinder)
        slide_raw_probabilities.append(raw_per_class_activations)
        slide_per_class_probabilities.append(per_class_probabilities)
        slide_predicted_argmax.append(predicted_argmax)

    modelNum = 0
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        slideHistograms = []
        simpleAccuracies = []
        model_slide_raw_probabilities = []
        model_slide_per_class_probabilities = []
        model_slide_predicted_argmax = []

        discriminativePatchFinder.setThreshold(th)
        discriminativePatchFinder.setThreshold(th)
        for i in range(len(slide_predicted_argmax)):
            if discriminativePatchFinder is not None:
                if discriminativePatchFinder.useDuringPredict():
                    print("Filtering discriminative patches for prediction")
                    H, _ = discriminativePatchFinder.filterDiscriminativePatches(
                        np.asarray(slide_per_class_probabilities[i]),
                        slide_predicted_argmax[i])
                    indices_to_ignore = []
                    for j in range(len(slide_predicted_argmax[i])):
                        if H[j] == 0:
                            indices_to_ignore.append(j)
                    model_slide_raw_probabilities.append(
                        np.delete(slide_raw_probabilities[i], indices_to_ignore, axis=0))
                    model_slide_per_class_probabilities.append(
                        np.delete(slide_per_class_probabilities[i], indices_to_ignore, axis=0))
                    model_slide_predicted_argmax.append(
                        np.delete(np.asarray(slide_predicted_argmax[i]), indices_to_ignore, axis=0))

            simpleAccuracy = accuracy_score([valSlideData.getSlideLabelList()[i]] * len(model_slide_predicted_argmax[i]), list(
                map(valSlideData.getLabelEncoder().inverse_transform, model_slide_predicted_argmax[i])))
            simpleAccuracies.append(simpleAccuracy)

            histogram = predict.histogram_for_predictions(model_slide_predicted_argmax[i])
            slideHistograms.append(histogram)

        maxPredictions = list(map(valSlideData.getLabelEncoder().inverse_transform, list(map(np.argmax, slideHistograms))))
        maxAccuracy = accuracy_score(valSlideData.getSlideLabelList(), maxPredictions)
        # print(accuracy)
        maxConfusionMatrix = confusion_matrix(valSlideData.getSlideLabelList(), maxPredictions)
        print("Max Confusion Matrix:\n %s" % maxConfusionMatrix)


        if (logRegModel[modelNum] is not None):
            logregAccuracy, logregConfusionMatrix = train_logreg.test_given_logreg(slideHistograms, valSlideData.getSlideLabelList(), logRegModel[modelNum])
            util.writeScalarSummary(logregAccuracy, "logRegAccuracyVal_th-" + str(th), netAccess.getSummmaryWriter(runName, sess.graph), step=step)
            print("LogReg Confusion Matrix:\n %s" % logregConfusionMatrix)

        # scalar, scalarName, summaryWriter, step
        util.writeScalarSummary(sum(simpleAccuracies) / valSlideData.getNumberOfSlides(), "simpleAccuracyVal_th-" + str(th), netAccess.getSummmaryWriter(runName, sess.graph),
                                step=step)
        util.writeScalarSummary(maxAccuracy, "maxAccuracyVal_th-" + str(th), netAccess.getSummmaryWriter(runName, sess.graph), step=step)

        netAccess.getSummmaryWriter(runName, sess.graph).flush()
        modelNum = modelNum + 1

    slideHistograms = []
    simpleAccuracies = []
    for i in range(valSlideData.getNumberOfSlides()):
        simpleAccuracy = accuracy_score([valSlideData.getSlideLabelList()[i]] * len(slide_predicted_argmax[i]), list(
            map(valSlideData.getLabelEncoder().inverse_transform, slide_predicted_argmax[i])))
        simpleAccuracies.append(simpleAccuracy)
        histogram = predict.histogram_for_predictions(slide_predicted_argmax[i])
        slideHistograms.append(histogram)

    maxPredictions = list(
        map(valSlideData.getLabelEncoder().inverse_transform, list(map(np.argmax, slideHistograms))))
    maxAccuracy = accuracy_score(valSlideData.getSlideLabelList(), maxPredictions)
    # print(accuracy)
    maxConfusionMatrix = confusion_matrix(valSlideData.getSlideLabelList(), maxPredictions)
    print("Max Confusion Matrix:\n %s" % maxConfusionMatrix)

    if (fullLogreg is not None):
        logregAccuracy, logregConfusionMatrix = train_logreg.test_given_logreg(slideHistograms,
                                                                               valSlideData.getSlideLabelList(),
                                                                               fullLogreg)
        util.writeScalarSummary(logregAccuracy, "logRegAccuracyVal",
                                netAccess.getSummmaryWriter(runName, sess.graph), step=step)
        print("LogReg Confusion Matrix:\n %s" % logregConfusionMatrix)

    # scalar, scalarName, summaryWriter, step
    util.writeScalarSummary(sum(simpleAccuracies) / valSlideData.getNumberOfSlides(),
                            "simpleAccuracyVal", netAccess.getSummmaryWriter(runName, sess.graph),
                            step=step)
    util.writeScalarSummary(maxAccuracy, "maxAccuracyVal",
                            netAccess.getSummmaryWriter(runName, sess.graph), step=step)

    netAccess.getSummmaryWriter(runName, sess.graph).flush()
