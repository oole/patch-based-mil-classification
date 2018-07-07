import validate
import predict
import tensorflow as tf
import data_tf
import netutil

def evaluateNet(netAccess: netutil.NetAccess, valSlideList: data_tf.SlideData):

    for slide in valSlideList.getSlideList():
        slide_y_pred, slide_y_pred_prob, slide_y_pred_argmax = predict.predict_given_net(pred_iterator_handle,
                                                                                         pred_iterator_len, netAccess,
                                                                                         batch_size=batchSize,
                                                                                         dropout_ratio=dropout_ratio,
                                                                                         sess=sess)
