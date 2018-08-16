from netutil import weight_variable, batch_norm, maxpool, conv2d, NetAccess
import tensorflow as tf

DEBUG = False

def tfPrint(tensor, prefix):
    return tf.Print(tensor,
                    [tf.shape(tensor)],
                    prefix)


def getLymphNet(scope, x, y,
                use_bn_1=True,
                use_bn_2=True,
                use_dropout_1=True,
                use_dropout_2=True,
                batchSize=64):

    with tf.variable_scope(scope):
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        is_training = tf.placeholder(tf.bool, name='phase')
        keep_prob = tf.placeholder(tf.float32)


        W_conv1 = weight_variable([5, 5, 3, 80], 'W_conv1')
        conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=[1, 1, 1, 1]))
        if DEBUG:
            conv1 = tfPrint(conv1,
                            "-----------------\n"
                            "Net:\n"
                            "conv1: ")
        if use_bn_1:

            bn1 = batch_norm(conv1, is_training, name='bn1')
            mp1 = maxpool(bn1, kernelSize=4, strideSize=2)
        else:
            mp1 = maxpool(conv1, kernelSize=4, strideSize=2)
        if DEBUG:
            mp1  = tfPrint(mp1,
                           "mp1: ")

        W_conv2 = weight_variable([3, 3, 80, 120], "W_conv2")
        conv2 = tf.nn.relu(conv2d(mp1, W_conv2))
        if DEBUG:
            conv2 = tfPrint(conv2,
                        "conv2: ")

        if use_bn_2:
            bn2 = batch_norm(conv2, is_training, name='bn2')
            mp2 = maxpool(bn2, kernelSize=3, strideSize=2)
        else:
            mp2 = maxpool(conv2, kernelSize=3, strideSize=2)
        if DEBUG:
            mp2 = tfPrint(mp2,
                        "mp2: ")

        W_conv3 = weight_variable([3, 3, 120, 160], "W_conv3")
        conv3 = tf.nn.relu(conv2d(mp2, W_conv3))
        if DEBUG:
            conv3 = tfPrint(conv3,
                      "conv3: ")

        W_conv4 = weight_variable([3, 3, 160, 200], "W_conv4")
        conv4 = tf.nn.relu(conv2d(conv3, W_conv4))
        if DEBUG:
            conv4 = tfPrint(conv4,
                        "conv4: ")

        mp3 = maxpool(conv4, kernelSize=3, strideSize=2)
        if DEBUG:
            mp3 = tfPrint(mp3,
                      "mp3: ")

        flat = tf.reshape(mp3, [-1, 4 * 4 * 200])
        if DEBUG:
            flat = tfPrint(flat,
                       "flatten: ")

        W_fc1 = weight_variable([4 * 4 * 200, 320], "W_fc1")
        fc1 = tf.nn.relu(tf.matmul(flat, W_fc1))

        if (use_dropout_1 == True):
            drop1 = tf.nn.dropout(fc1, keep_prob)

            W_fc2 = weight_variable([320, 320], "W_fc2")
            fc2 = tf.nn.relu(tf.matmul(drop1, W_fc2))
        else:
            W_fc2 = weight_variable([320, 320], "W_fc2")
            fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2))

        if (use_dropout_2 == True):

            drop2 = tf.nn.dropout(fc2, keep_prob)

            W_fc3 = weight_variable([320, 3], "W_fc3")
            y_pred = tf.matmul(drop2, W_fc3)
        else:
            W_fc3 = weight_variable([320, 3], "W_fc3")
            y_pred = tf.matmul(fc2, W_fc3)


        # Prediction probabilities for classes
        y_pred_prob = (tf.nn.softmax(y_pred))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))
        learning_rate = tf.placeholder(tf.float32)

        train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=globalStep)

        y_argmax = tf.argmax(y_pred, 1)
        y_argmax_given = tf.argmax(y, 1)
        correct_prediction = tf.equal(y_argmax, y_argmax_given)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        image_summary_t = tf.cond(is_training, lambda: tf.summary.image("Training", x, max_outputs=1),
                                  lambda: tf.summary.image("Testing", x, max_outputs=1))
        # image_summary_t = tf.summary.image("Training", x, max_outputs=1)

        return NetAccess(train, loss, y, accuracy, x, image_summary_t, keep_prob, learning_rate, is_training, y_pred,
                         y_argmax, y_pred_prob, globalStep, batchSize=batchSize)
