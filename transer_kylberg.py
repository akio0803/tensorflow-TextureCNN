# coding: utf-8

from scipy.misc import imread
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import tensorflow as tf
import numpy as np
import cPickle as pickle
import datetime
import argparse

from include.layer import TCNN_Alex

parser = argparse.ArgumentParser()
parser.add_argument('--test_size', default=0.2)
parser.add_argument('--train_loop', default=500)
parser.add_argument('--trial_num', default=50)
parser.add_argument('--gpu', default=0)
args = parser.parse_args()


today = datetime.datetime.today().strftime("%m%d_%H_%M")
test_size = float(args.test_size)
train_loop = int(args.train_loop)
trial_num = int(args.trial_num)
gpu = int(args.gpu)


def get_jpg_data(path):
    def extract_image(path):
        img = imread(path)
        img = img / 255.0
        return img

    file_name_list = os.listdir(path)
    file_data_list = [extract_image(path+file_path) for file_path in file_name_list]

    return np.array(file_data_list)


def arrange_data(data_list):
    train_X = np.empty((0, 227, 227, 3))
    test_X = np.empty((0, 227, 227, 3))
    train_y = np.empty((0, 1))
    test_y = np.empty((0, 1))
    for label, data in data_list:
        test_label = np.array([label] * len(data)).reshape(-1, 1)
        _train_X, _test_X, _train_y, _test_y = train_test_split(data, test_label, test_size=test_size)
        train_X = np.concatenate((train_X, _train_X), axis=0)
        test_X = np.concatenate((test_X, _test_X), axis=0)
        train_y = np.concatenate((train_y, _train_y), axis=0)
        test_y = np.concatenate((test_y, _test_y), axis=0)

    train_X, train_y = shuffle(train_X, train_y)
    test_X, test_y = shuffle(test_X, test_y)
    return train_X.astype('float32'), test_X.astype('float32'), train_y, test_y


def data_preprocess():
    SM_path_list = [
        '/data/unagi0/hayakawa/keio/data/data20160223/image/SM massive/227_227/',
        '/data/unagi0/hayakawa/keio/data/data20161007/image/SM massive/227_227/'
        ]
    M_path_list = [
        '/data/unagi0/hayakawa/keio/data/data20160223/image/adenoma M/227_227/',
        '/data/unagi0/hayakawa/keio/data/data20161007/image/adenoma M/227_227/'
        ]

    ## tf input -> [batch, height, width, channel]
    SM_data_list = []
    for SM_path in SM_path_list:
        SM_data_list.append((0, get_jpg_data(SM_path)))

    M_data_list = []
    for M_path in M_path_list:
        M_data_list.append((1, get_jpg_data(M_path)))

    data_list = SM_data_list + M_data_list

    train_X, test_X, train_y, test_y = arrange_data(data_list)

    return train_X, test_X, train_y, test_y


def run_process(trial=trial_num):
    with tf.device('/gpu:{}'.format(gpu)):
        # define network
        x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='input')
        keep_prob = tf.placeholder(tf.float32, name='no_dropout_rate')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='label')

        network = TCNN_Alex(keep_prob, out_class=2, is_BN=True)
        fc3 = network.build(x)
        parameters = network.get_params()

        # for inference
        pred_y = tf.nn.softmax(fc3)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(fc3, tf.one_hot(tf.cast(y, tf.int32), 2))
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("entropy", loss)

        optimizer = tf.train.AdamOptimizer()

        train_step = optimizer.minimize(loss, var_list=parameters)

        eval_prediction = tf.nn.in_top_k(pred_y, tf.reshape(tf.cast(y, tf.int32), [-1]), 1)
        accuracy = tf.reduce_mean(tf.cast(eval_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    result = []
    for cnt in range(trial):
        train_X, test_X, train_y, test_y = data_preprocess()

        train_summary_writer = tf.train.SummaryWriter('output/transfer_kylberg/{}/{}/train/'.format(today, cnt))
        test_summary_writer = tf.train.SummaryWriter('output/transfer_kylberg/{}/{}/test/'.format(today, cnt))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()

            # restore
            restored_parameters = parameters[:-3]

            restore_saver = tf.train.Saver(restored_parameters)

            ckpt = tf.train.get_checkpoint_state('/data/unagi0/hayakawa/keio/code_tf/model/kylberg')
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring model from {}'.format(ckpt.model_checkpoint_path))
                restore_saver.restore(sess, ckpt.model_checkpoint_path)

            saver = tf.train.Saver(parameters)

            batch_size = 64
            test_acc_list = []
            print "======== trial: {} ========".format(cnt)
            for loop in range(train_loop + 1):
                train_loss_list = []
                train_acc_list = []
                for i in range(train_X.shape[0] // batch_size + 1):
                    start = i * batch_size
                    stop = min(start + batch_size, len(train_X))
                    batch_x = train_X[start:stop]
                    batch_y = train_y[start:stop]
                    train_summary, _, loss_value, acc_value = sess.run([merged, train_step, loss, accuracy],
                                                                       feed_dict={x: batch_x,
                                                                                  y: batch_y,
                                                                                  keep_prob: 0.5})
                    train_loss_list.append(loss_value)
                    train_acc_list.append(acc_value)
                if loop % 50 == 0:
                    train_summary_writer.add_summary(train_summary, loop)

                    test_summary, test_acc = sess.run([merged, accuracy],
                                                      feed_dict={x: test_X, y: test_y, keep_prob: 1.})
                    test_summary_writer.add_summary(test_summary, loop)

                    test_acc_list.append(test_acc)

                    print "------ loop: {} ------".format(loop)
                    print "mean training loss {}".format(np.array(train_loss_list).mean())
                    print "mean training accuracy {}".format(np.array(train_acc_list).mean())
                    print "test accuracy: {}".format(test_acc)

            os.mkdir("model/transfer_kylberg/{}/{}".format(today, cnt))
            saver.save(sess, "model/transfer_kylberg/{}/{}/TCNN_scratch.ckpt".format(today, cnt))

            result.append(test_acc_list)

    return result


if __name__ == '__main__':
    if os.path.exists("model/transfer_kylberg/{}".format(today)) is False:
        os.makedirs("model/transfer_kylberg/{}".format(today))

    result = run_process(trial=trial_num)

    with open("output/transfer_kylberg/{}/result.pickle" .format(today), "wb") as f:
        pickle.dump(result, f)
