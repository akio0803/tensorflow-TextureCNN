# coding: utf-8
import os
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from multiprocessing import Pool

from include.layer import TCNN_Alex
from sklearn.utils import shuffle


data_path = '/data/unagi0/hayakawa/keio/data/kylberg/original/data/'
out_dir_path = '/data/unagi0/hayakawa/keio/data/kylberg/preprocess/'
model_out = '/data/unagi0/hayakawa/keio/code_tf/model/kylberg/kylberg_2.ckpt'
train_loop = 500


def get_kylberg_data(class_name):
    path = os.path.join(out_dir_path, class_name)

    result = []
    for img_name in os.listdir(path):
        img = imread(os.path.join(path, img_name))
        img = img / 255.
        img = np.tile(img, (3, 1, 1))  # convert gray to rgb

        result.append(img)

    return (class_name, np.array(result))


def data_preprocess():

    p = Pool(6)
    load_data = p.map(get_kylberg_data, os.listdir(out_dir_path))

    data = np.array([x for _, x in load_data])

    class_num = len(data)
    img_num = len(data[0])
    label = np.tile(np.arange(class_num).reshape(-1, 1), img_num).reshape(-1, 1)

    data = data.reshape(-1, 227, 227, 3)

    data, label = shuffle(data, label)

    return data.astype("float32"), label.astype("float32"), class_num


if __name__ == '__main__':
    # preprocess
    print "--- start data preprocess ---"
    data, label, class_num = data_preprocess()

    with tf.device('/gpu:1'):
        x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='input')
        keep_prob = tf.placeholder(tf.float32, name='no_dropout_rate')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='label')

        network = TCNN_Alex(keep_prob, out_class=class_num, is_BN=True)
        fc3 = network.build(x)
        parameters = network.get_params()
        print parameters == tf.trainable_variables()

        # for inference
        pred_y = tf.nn.softmax(fc3)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(fc3, tf.one_hot(tf.cast(y, tf.int32), class_num))
        loss = tf.reduce_mean(cross_entropy)
        s1 = tf.summary.scalar("entropy", loss)

        optimizer = tf.train.AdamOptimizer()

        train_step = optimizer.minimize(loss, var_list=parameters)

        eval_prediction = tf.nn.in_top_k(pred_y, tf.reshape(tf.cast(y, tf.int32), [-1]), 1)
        accuracy = tf.reduce_mean(tf.cast(eval_prediction, tf.float32))
        s2 = tf.summary.scalar("accuracy", accuracy)

    summary_writer = tf.train.SummaryWriter('output/kylberg')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge([s1, s2])
        saver = tf.train.Saver()

        batch_size = 64
        for loop in range(train_loop):
            train_loss_list = []
            train_acc_list = []
            for i in range(data.shape[0] // batch_size):
                start = i * batch_size
                stop = min(start + batch_size, len(data))
                batch_x = data[start:stop]
                batch_y = label[start:stop]
                summary, _, loss_value, acc_value = sess.run([merged, train_step, loss, accuracy],
                                                             feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        keep_prob: 0.5})
                train_loss_list.append(loss_value)
                train_acc_list.append(acc_value)
            if loop % 10 == 0:
                summary_writer.add_summary(summary, loop)

                print "------ loop: %d ------" % loop
                print "mean training loss %g" % np.array(train_loss_list).mean()
                print "mean training accuracy %g" % np.array(train_acc_list).mean()

        saver.save(sess, model_out)
