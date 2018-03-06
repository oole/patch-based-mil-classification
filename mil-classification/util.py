import tensorflow as tf

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x,b)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

def conv_net(weights, biases, dropout):
    x = tf.placeholder(tf.float32, shape=[None, 400, 400, 3])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
