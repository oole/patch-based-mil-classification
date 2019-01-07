from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import data_tf
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
import predict

def train_logreg(netAccess, savepath, trainSlideData, dropout,  sess, discriminativePatchFinder=None):
    slideHistograms = []
    iterator, iteratorInitOps = trainSlideData.getIterator(netAccess, augment=True)
    for i in range(trainSlideData.getNumberOfSlides()):
        slideIteratorLen = len(trainSlideData.getSlideList()[i])
        sess.run(iteratorInitOps[i])
        slideIteratorHandle = sess.run(iterator.string_handle())
        slide_y_pred, slide_y_pred_prob, slide_y_pred_argmax = \
            predict.predict_given_net(slideIteratorHandle, slideIteratorLen,
                                      netAccess, batch_size=netAccess.getBatchSize(), dropout_ratio=dropout, sess=sess,
                                      discriminativePatchFinder=discriminativePatchFinder)
        histogram = predict.histogram_for_predictions(slide_y_pred_argmax)
        slideHistograms.append(histogram)

    logregModel = train_logreg_from_histograms_and_labels(slideHistograms, trainSlideData.getSlideLabelList(), savepath)
    return logregModel

def train_logreg_from_histograms_and_labels(histograms, labels, savepath=None):
    model = LogisticRegression(verbose=1)
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
    print (accuracy)
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


