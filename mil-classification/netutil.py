import tensorflow as tf


def weight_variable(shape, name, trainable = True, zero = False):
    weight_decay = tf.constant(0.001, dtype=tf.float32)
    if zero:
        return tf.get_variable(name=name, shape=shape, initializer = tf.constant_initializer(0), trainable=trainable)
    else:
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer(),
                               trainable=trainable, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

def bias_variable(shape, name, trainable=True, zero=False):
    if zero:
        return tf.get_variable(name=name, shape=shape, initializer = tf.constant_initializer(0), trainable=trainable)
    else:
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1), trainable=trainable)

def batch_norm(x, is_training, name=None, trainable=True):
    bn = tf.layers.batch_normalization(x, momentum=0.9, training=is_training, name=name, trainable=trainable)
    return bn

def conv2d(x, w, stride=[1,1,1,1], padding="VALID"):
    return tf.nn.conv2d(x, filter=w, strides=stride, padding=padding)

def maxpool(x, strideSize=2, kernelSize=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1,kernelSize,kernelSize,1], strides=[1,strideSize,strideSize,1], padding=padding)


def build_model(scope, x, y,
                use_bn_1=True,
                use_bn_2=True,
                use_dropout_1=True,
                use_dropout_2=True):
    with tf.variable_scope(scope):
        is_training = tf.placeholder(tf.bool, name='phase')
        keep_prob = tf.placeholder(tf.float32)

        #x = tf.placeholder(tf.float32, shape=[None, 400,400,3])

        W_conv1 = weight_variable([10,10,3,80], 'W_conv1')
        conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=[1,2,2,1]))

        if use_bn_1:

            bn1 = batch_norm(conv1, is_training, name='bn1')
            mp1 = maxpool(bn1, kernelSize=6, strideSize=4)
        else:
            mp1 = maxpool(conv1, kernelSize=6, strideSize=4)

        W_conv2 = weight_variable([5,5,80,120], "W_conv2")
        conv2 = tf.nn.relu(conv2d(mp1, W_conv2))


        if use_bn_2:
            bn2 = batch_norm(conv2, is_training, name='bn2')
            mp2 = maxpool(bn2, kernelSize=3, strideSize=2)
        else:
            mp2 = maxpool(conv2, kernelSize=3, strideSize=2)

        W_conv3 = weight_variable([3,3,120,160], "W_conv3")
        conv3 = tf.nn.relu(conv2d(mp2, W_conv3))

        W_conv4 = weight_variable([3,3,160,200], "W_conv4")
        conv4 = tf.nn.relu(conv2d(conv3, W_conv4))

        mp3 = maxpool(conv4, kernelSize=3, strideSize=2)

        flat = tf.reshape(mp3, [-1,8*8*200])

        W_fc1 = weight_variable([8*8*200, 320], "W_fc1")
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

            W_fc3 = weight_variable([320,6], "W_fc3")
            y_pred = tf.matmul(drop2, W_fc3)
        else:
            W_fc3 = weight_variable([320, 6], "W_fc3")
            y_pred = tf.matmul(fc2, W_fc3)

        #y = tf.placeholder(tf.uint8, [None, 6])

        # Prediction probabilities for classes
        y_pred_prob = (tf.nn.softmax(y_pred))

        # one_hot_y = tf.one_hot(y, 6)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits = y_pred))
        learning_rate = tf.placeholder(tf.float32)

        train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        y_argmax = tf.argmax(y_pred, 1)
        correct_prediction = tf.equal(y_argmax, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return train, loss, y, accuracy, x, keep_prob, learning_rate, is_training, y_pred, y_argmax, y_pred_prob



# train, loss, y, accuracy, x, keep_prob, learning_rate = build_model('lol')