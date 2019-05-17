from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import data_tf
from sklearn.metrics import accuracy_score, confusion_matrix
import predict
import numpy as np


def train_logreg(netAccess, savepath, trainSlideData, dropout, sess, discriminativePatchFinder=None):
    iterator, iteratorInitOps = trainSlideData.getIterator(netAccess, augment=True)
    slide_raw_probabilities = []
    slide_per_class_probabilities = []
    slide_predicted_argmax = []
    for i in range(trainSlideData.getNumberOfSlides()):
        slideIteratorLen = len(trainSlideData.getSlideList()[i])
        sess.run(iteratorInitOps[i])
        slideIteratorHandle = sess.run(iterator.string_handle())
        raw_per_class_activations, per_class_probabilities, predicted_argmax = \
            predict.predict_given_net(slideIteratorHandle, slideIteratorLen,
                                      netAccess, batch_size=netAccess.getBatchSize(), dropout_ratio=dropout, sess=sess,
                                      discriminativePatchFinder=discriminativePatchFinder)
        slide_raw_probabilities.append(raw_per_class_activations)
        slide_per_class_probabilities.append(per_class_probabilities)
        slide_predicted_argmax.append(predicted_argmax)

    # filtered logregModels
    logRegModelList = []
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        slideHistograms = []
        model_slide_raw_probabilities = []
        model_slide_per_class_probabilities = []
        model_slide_predicted_argmax = []

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

            histogram = predict.histogram_for_predictions(model_slide_predicted_argmax[i])
            slideHistograms.append(histogram)
        if savepath is not None:
            logsavePath = savepath + "_th-" + str(th)
        else:
            logsavePath = savepath
        logregModel = train_logreg_from_histograms_and_labels(slideHistograms, trainSlideData.getSlideLabelList(),
                                                              logsavePath)
        logRegModelList.append(logregModel)

    # full logreModels
    fullSlideHistograms = []
    for i in range(len(slide_predicted_argmax)):
        histogram = predict.histogram_for_predictions(slide_predicted_argmax[i])
        fullSlideHistograms.append(histogram)

    fullModel = train_logreg_from_histograms_and_labels(fullSlideHistograms, trainSlideData.getSlideLabelList(),
                                                        logsavePath)
    return logRegModelList, fullModel


def train_logreg_from_histograms_and_labels(histograms, labels, savepath=None):
    model = LogisticRegression(verbose=1)
    print("#Histogram + %s, #labels + %s" % (str(len(histograms)), str(len(labels))))
    model.fit(histograms, labels)
    print(model)
    print("train_logreg, finished")
    if savepath is not None:
        joblib.dump(model, savepath)
        print("train_logreg, saved model to:  " + savepath)
    return model


def save_logreg_model(model, savepath):
    joblib.dump(model, savepath)
    print("train_logreg, saved model to:  " + savepath)


def test_logreg(datapath, loadpath):
    histograms, labels = data_tf.read_histograms_and_labels_from_file(datapath)
    model = joblib.load(loadpath)
    predictions = model.predict(histograms)
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    confusion = confusion_matrix(labels, predictions)
    print(confusion)


def test_given_logreg(histograms, labels, logreg_model):
    predictions = logreg_model.predict(histograms)
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    confusion = confusion_matrix(labels, predictions)
    print(confusion)
    return accuracy, confusion


def test_logreg2(histograms, labels, loadpath):
    model = joblib.load(loadpath)
    predictions = model.predict(histograms)
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    confusion = confusion_matrix(labels, predictions)
    print(confusion)
    return accuracy, confusion
