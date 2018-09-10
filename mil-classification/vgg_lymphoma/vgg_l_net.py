from netutil import weight_variable, batch_norm, maxpool, conv2d, NetAccess
import tensorflow as tf

DEBUG = False

def tfPrint(tensor, prefix):
    return tf.Print(tensor,
                    [tf.shape(tensor)],
                    prefix)

BLOCKNUMBER = 1

def build_block(input, shape, convnums):
    global BLOCKNUMBER
    convNum = 1
    blockname = "block" + str(BLOCKNUMBER) + "_"

    W_conv = weight_variable(shape, blockname + 'W_conv' + str(convNum))
    conv = tf.nn.relu(conv2d(input, W_conv, stride=[1, 1, 1, 1], padding="SAME"))
    convNum += 1

    for i in range(convnums-1):
        W_conv = weight_variable([shape[0], shape[1], shape[3], shape[3]], blockname + 'W_conv' + str(convNum))
        conv = tf.nn.relu(conv2d(conv, W_conv, stride=[1, 1, 1, 1], padding="SAME"))
        convNum += 1

    mp = maxpool(conv, kernelSize=2, strideSize=2, name=blockname + 'pool')

    BLOCKNUMBER += 1

    return mp



def getLymphNet(scope, x, y,
                use_bn_1=True,
                use_bn_2=True,
                use_dropout_1=True,
                use_dropout_2=True,
                batchSize=64):
    global BLOCKNUMBER
    BLOCKNUMBER = 1
    with tf.variable_scope(scope):
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        is_training = tf.placeholder(tf.bool, name='phase')
        keep_prob = tf.placeholder(tf.float32)

        block_1 = build_block(x, [3,3,3,64], 2)
        block_2 = build_block(block_1, [3,3,64,128], 2)
        block_3 = build_block(block_2, [3,3,128,256], 3)
        block_4 = build_block(block_3, [3,3,256,512], 3)
        block_5 = build_block(block_4, [3,3,512,512], 3)
        block_5_shape = block_5.get_shape().as_list()


        flat = tf.reshape(block_5, [-1, block_5_shape[1] * block_5_shape[2] * block_5_shape[3]])

        W_fc1 = weight_variable([block_5_shape[1] * block_5_shape[2] * block_5_shape[3], 64], name="dense1")
        fc1 = tf.nn.relu(tf.matmul(flat, W_fc1))
        drop1 = tf.nn.dropout(fc1, keep_prob, name="drop1")
        W_fc2 = weight_variable([64, 3], name="dense2")
        y_pred = tf.matmul(drop1, W_fc2)


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

# getLymphNet("testscope", tf.placeholder(dtype=tf.float32,shape=[None, 64,64, None]), tf.placeholder(dtype=tf.float32,shape=[None, 3]))