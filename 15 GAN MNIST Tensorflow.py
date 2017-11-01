import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    # return tf.random_normal(shape=size, stddev=xavier_stddev)
    return xavier_stddev


def conv(x, w, b, stride, name):
    with tf.variable_scope('conv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding='SAME',
                           name=name) + b


def deconv(x, w, b, shape, stride, name):
    with tf.variable_scope('deconv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d_transpose(x,
                                       filter=w,
                                       output_shape=shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME',
                                       name=name) + b


def lrelu(x, alpha=0.2):
    with tf.variable_scope('leakyReLU'):
        return tf.maximum(x, alpha * x)


def discriminator(X, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        K = 64
        M = 128
        N = 256

        W1 = tf.get_variable('D_W1', [4, 4, 1, K], initializer=tf.random_normal_initializer(stddev=0.1))
        B1 = tf.get_variable('D_B1', [K], initializer=tf.constant_initializer())
        W2 = tf.get_variable('D_W2', [4, 4, K, M], initializer=tf.random_normal_initializer(stddev=0.1))
        B2 = tf.get_variable('D_B2', [M], initializer=tf.constant_initializer())
        W3 = tf.get_variable('D_W3', [7*7*M, N], initializer=tf.random_normal_initializer(stddev=0.1))
        B3 = tf.get_variable('D_B3', [N], initializer=tf.constant_initializer())
        W4 = tf.get_variable('D_W4', [N, 1], initializer=tf.random_normal_initializer(stddev=0.1))
        B4 = tf.get_variable('D_B4', [1], initializer=tf.constant_initializer())

        X = tf.reshape(X, [-1, 28, 28, 1], 'reshape')

        conv1 = conv(X, W1, B1, stride=2, name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1)
        conv2 = conv(tf.nn.dropout(lrelu(bn1), 0.4), W2, B2, stride=2, name='conv2')
        # conv2 = conv(lrelu(conv1), W2, B2, stride=2, name='conv2')

        bn2 = tf.contrib.layers.batch_norm(conv2)
        flat = tf.reshape(tf.nn.dropout(lrelu(bn2), 0.4), [-1, 7*7*M], name='flat')
        # flat = tf.reshape(lrelu(conv2), [-1, 7*7*M], name='flat')

        dense = lrelu(tf.matmul(flat, W3) + B3)
        logits = tf.matmul(dense, W4) + B4
        prob = tf.nn.sigmoid(logits)
        return prob, logits


def generator(X, batch_size=64):
    with tf.variable_scope('generator'):

        K = 256
        L = 128
        M = 64

        W1 = tf.get_variable('G_W1', [100, 7*7*K], initializer=tf.random_normal_initializer(stddev=0.1))
        B1 = tf.get_variable('G_B1', [7*7*K], initializer=tf.constant_initializer())

        W2 = tf.get_variable('G_W2', [4, 4, M, K], initializer=tf.random_normal_initializer(stddev=0.1))
        B2 = tf.get_variable('G_B2', [M], initializer=tf.constant_initializer())

        W3 = tf.get_variable('G_W3', [4, 4, 1, M], initializer=tf.random_normal_initializer(stddev=0.1))
        B3 = tf.get_variable('G_B3', [1], initializer=tf.constant_initializer())

        X = lrelu(tf.matmul(X, W1) + B1)
        X = tf.reshape(X, [batch_size, 7, 7, K])
        deconv1 = deconv(X, W2, B2, shape=[batch_size, 14, 14, M], stride=2, name='deconv1')
        bn1 = tf.contrib.layers.batch_norm(deconv1)
        deconv2 = deconv(tf.nn.dropout(lrelu(bn1), 0.4), W3, B3, shape=[batch_size, 28, 28, 1], stride=2, name='deconv2')

        XX = tf.reshape(deconv2, [-1, 28*28], 'reshape')

        return tf.nn.sigmoid(XX)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse


def read_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    return mnist


def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def train(logdir, batch_size):
    
    mnist = read_data()

    with tf.variable_scope('placeholder'):
        # Raw image
        X = tf.placeholder(tf.float32, [None, 784])
        tf.summary.image('raw image', tf.reshape(X, [-1, 28, 28, 1]), 3)
        # Noise
        z = tf.placeholder(tf.float32, [None, 100])  # noise
        tf.summary.histogram('Noise', z)

    with tf.variable_scope('GAN'):
        G = generator(z, batch_size)

        D_real, D_real_logits = discriminator(X, reuse=False)
        D_fake, D_fake_logits = discriminator(G, reuse=True)
    tf.summary.image('generated image', tf.reshape(G, [-1, 28, 28, 1]), 3)

    with tf.variable_scope('Prediction'):
        tf.summary.histogram('real', D_real)
        tf.summary.histogram('fake', D_fake)

    with tf.variable_scope('D_loss'):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        d_loss = d_loss_real + d_loss_fake

        tf.summary.scalar('d_loss_real', d_loss_real)
        tf.summary.scalar('d_loss_fake', d_loss_fake)
        tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('G_loss'):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                (logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
        tf.summary.scalar('g_loss', g_loss)

    tvar = tf.trainable_variables()
    dvar = [var for var in tvar if 'discriminator' in var.name]
    gvar = [var for var in tvar if 'generator' in var.name]

    with tf.name_scope('train'):
        d_train_step = tf.train.AdamOptimizer().minimize(d_loss, var_list=dvar)
        g_train_step = tf.train.AdamOptimizer().minimize(g_loss, var_list=gvar)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/'+'gan_conv_'+logdir)
    writer.add_graph(sess.graph)

    num_img = 0
    if not os.path.exists('output/'):
        os.makedirs('output/')

    for i in range(100000):
        batch_X, _ = mnist.train.next_batch(batch_size)
        batch_noise = np.random.uniform(-1., 1., [batch_size, 100])

        if i % 500 == 0:
            samples = sess.run(G, feed_dict={z: np.random.uniform(-1., 1., [64, 100])})
            fig = plot(samples)
            plt.savefig('output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)

        _, d_loss_print = sess.run([d_train_step, d_loss],
                                   feed_dict={X: batch_X, z: batch_noise})

        _, g_loss_print = sess.run([g_train_step, g_loss],
                                   feed_dict={z: batch_noise})

        if i % 100 == 0:
            s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise})
            writer.add_summary(s, i)
            print('epoch:%d g_loss:%f d_loss:%f' % (i, g_loss_print, d_loss_print))



parser = argparse.ArgumentParser(description='Train vanila GAN using convolutional networks')
parser.add_argument('--logdir', type=str, default='1', help='logdir for Tensorboard, give a string')
parser.add_argument('--batch_size', type=int, default=64, help='batch size: give a int')
args = parser.parse_args()

train(logdir=args.logdir, batch_size=args.batch_size)

