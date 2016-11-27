# coding: utf-8
import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)


def batch_norm(x, eps=1e-4):
    x_csize = int(x.get_shape()[-1])
    beta = bias_variable(shape=[x_csize])
    gamma = weight_variable(shape=[x_csize])
    params = [beta, gamma]

    mean, variance = tf.nn.moments(x, axes=[0])
    normed = gamma * (x - mean) / tf.sqrt(variance + eps) + beta

    return normed, params

class AlexNet(object):
    def __init__(self, keep_prob, out_class, is_BN=True):
        self.parameters = []
        self.is_BN = is_BN
        self.keep_prob = keep_prob
        self.out_class = out_class

    def build(self, x):
        if self.is_BN is True:
            input_x, BN_params = batch_norm(x)
            self.parameters += BN_params
        else:
            input_x = x

        pool5 = self.build_conv(input_x)

        pool5_size = pool5.get_shape()
        pool5_flat = tf.reshape(pool5, [-1, pool5_size[1] * pool5_size[2] * pool5_size[3]])

        fc3 = self.build_fc(pool5_flat)

        return fc3

    def build_conv(self, input_x):
        pool1 = self.conv1(input_x)
        pool2 = self.conv2(pool1)
        pool3 = self.conv3(pool2)
        pool4 = self.conv4(pool3)
        pool5 = self.conv5(pool4)

        return pool5

    def build_fc(self, conv_out):
        fc1 = self.fc1(conv_out)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)

        return fc3

    def conv1(self, input_x):
        with tf.name_scope('conv1') as scope:
            kernel = weight_variable(shape=[11, 11, 3, 96], name='weight1')
            conv = tf.nn.conv2d(input_x, kernel, [1, 4, 4, 1], padding='SAME')
            self.parameters += [kernel]

            if self.is_BN is True:
                u, BN_params = batch_norm(conv)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[96], name='bias1')
                u = tf.nn.bias_add(conv, bias)
                self.parameters += [bias]

            conv1 = tf.nn.relu(u, name=scope)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')

        return pool1

    def conv2(self, pool1):
        with tf.name_scope('conv2') as scope:
            kernel = weight_variable(shape=[5, 5, 96, 256], name='weight2')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            self.parameters += [kernel]

            if self.is_BN is True:
                u, BN_params = batch_norm(conv)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[256], name='bias2')
                u = tf.nn.bias_add(conv, bias)
                self.parameters += [bias]

            conv2 = tf.nn.relu(u, name=scope)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')

        return pool2

    def conv3(self, pool2):
        with tf.name_scope('conv3') as scope:
            kernel = weight_variable(shape=[5, 5, 256, 384], name='weight3')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            self.parameters += [kernel]

            if self.is_BN is True:
                u, BN_params = batch_norm(conv)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[384], name='bias3')
                u = tf.nn.bias_add(conv, bias)
                self.parameters += [bias]

            conv3 = tf.nn.relu(u, name=scope)

        return conv3

    def conv4(self, pool3):
        with tf.name_scope('conv4') as scope:
            kernel = weight_variable(shape=[5, 5, 384, 384], name='weight4')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            self.parameters += [kernel]

            if self.is_BN is True:
                u, BN_params = batch_norm(conv)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[384], name='bias4')
                u = tf.nn.bias_add(conv, bias)
                self.parameters += [bias]

            conv4 = tf.nn.relu(u, name=scope)

        return conv4

    def conv5(self, pool4):
        with tf.name_scope('conv5') as scope:
            kernel = weight_variable(shape=[5, 5, 384, 256], name='weight5')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            self.parameters += [kernel]

            if self.is_BN is True:
                u, BN_params = batch_norm(conv)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[256], name='bias5')
                u = tf.nn.bias_add(conv, bias)
                self.parameters += [bias]

            conv5 = tf.nn.relu(u, name=scope)

        return conv5

    def fc1(self, conv_out):
        conv_size = conv_out.get_shape()
        reshape_size = int(conv_size[1])
        with tf.name_scope('fc1') as scope:
            weight = weight_variable(shape=[reshape_size, 4096], name='weight6')
            fc = tf.matmul(conv_out, weight)
            self.parameters += [weight]

            if self.is_BN is True:
                u, BN_params = batch_norm(fc)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[4096], name='bias6')
                u = tf.nn.bias_add(fc, bias)
                self.parameters += [bias]

            fc1 = tf.nn.relu(u)
            fc1_drop = tf.nn.dropout(fc1, self.keep_prob, name=scope)

        return fc1_drop

    def fc2(self, fc1):
        with tf.name_scope('fc2') as scope:
            weight = weight_variable(shape=[4096, 4096], name='weight7')
            fc = tf.matmul(fc1, weight)
            self.parameters += [weight]

            if self.is_BN is True:
                u, BN_params = batch_norm(fc)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[4096], name='bias7')
                u = tf.nn.bias_add(fc, bias)
                self.parameters += [bias]

            fc2 = tf.nn.relu(u)
            fc2_drop = tf.nn.dropout(fc2, self.keep_prob, name=scope)

        return fc2_drop

    def fc3(self, fc2):
        with tf.name_scope('fc3') as scope:
            weight = weight_variable(shape=[4096, self.out_class], name='weight8')
            fc = tf.matmul(fc2, weight)
            self.parameters += [weight]

            if self.is_BN is True:
                u, BN_params = batch_norm(fc)
                self.parameters += BN_params
            else:
                bias = bias_variable(shape=[self.out_class], name='bias8')
                u = tf.nn.bias_add(fc, bias)
                self.parameters += [bias]

            fc3 = u

        return fc3

    def get_params(self):
        return self.parameters

class TCNN_Alex(AlexNet):
    def __init__(self, keep_prob, out_class, is_BN=True):
        super(TCNN_Alex, self).__init__(keep_prob, out_class, is_BN)

    def build(self, x):
        if self.is_BN is True:
            input_x, BN_params = batch_norm(x)
            self.parameters += BN_params
        else:
            input_x = x

        pool5 = self.build_conv(input_x)

        powered = self.power_layer(pool5)
        powered_size = powered.get_shape()
        reshape_size = int(powered_size[1] * powered_size[2] * powered_size[3])
        powered_flat = tf.reshape(powered, [-1, reshape_size])

        fc3 = self.build_fc(powered_flat)

        return fc3


    def power_layer(self, x):
        x_size = x.get_shape()
        out = tf.nn.max_pool(x, ksize=[1, x_size[1], x_size[2], 1],
                               strides=[1, 1, 1, 1], padding='VALID')

        return out
