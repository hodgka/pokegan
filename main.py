import os

import numpy as np
import tensorflow as tf

from dataio import Dataset
from layers import *
from utils import pp, show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("iterations", 1000, "Epoch to train [1e5]")
flags.DEFINE_float("clipping", 0.01, "Clipping parameter c. [0.01]")
flags.DEFINE_integer("n_d", 5, "Number of iterations to train the discriminator. [5]")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate of for adam [1e-4]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_integer("input_height", 96, "The size of image to use (will be center cropped). [96]")
flags.DEFINE_integer("input_width", 96, "The size of image to use (will be center cropped). [96]")
flags.DEFINE_integer("channels", 3, "Number of channels. [3]")
flags.DEFINE_integer("g_train", 10, "Number of iterations to train the generator for. [5]")
flags.DEFINE_integer("output_height", 96, "The size of the output images to produce [96]")
flags.DEFINE_integer("output_width", 96, "The size of the output images to produce. [96]")
flags.DEFINE_string("dataset", "pokemon", "The name of dataset [pokemon]")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*.png]")
flags.DEFINE_string("model_dir", "fully_connected", "Directory name to save the checkpoints [fully_connected]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

tf.set_random_seed(0)
np.random.seed(0)
pp.pprint(FLAGS.__flags)


def generator(x, reuse=False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        net = x  # batch_size x 100
        net = linear(net, 512 * 16, name="genf1")
        net = linear(net, 256 * 8, name="genf2")
        net = linear(net, 128 * 8, name="genf3")
        net = linear(net, 96 * 96 * 3, name="genf4", bn=False)
        net = tf.nn.tanh(net)
        net = tf.reshape(net, (-1, 96, 96, 3))
        # net = tf.reshape(net, [-1, 4, 4, 1024])
        # net = deconv_layer(net, 512, (3, 3), (2, 2), "SAME", "gd1")  # 8x8x512
        # net = deconv_layer(net, 256, (5, 5), (2, 2), "SAME", "gd2")  # 16x16x256
        # net = deconv_layer(net, 128, (5, 5), (2, 2), "SAME", "gd3")  # 32x32x128
        # net = deconv_layer(net,   3, (5, 5), (2, 2), "SAME", "gd4", activation=tf.nn.tanh)  # 96x96x3

        return net


def discriminator(x, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.reshape(x, (-1, 96 * 96 * 3))
        # net = linear(net, 128, name="df2")
        net = linear(net, 256, name="discf1")
        net = linear(net, 100, name="discf2")
        net = linear(net, 100, name="discf3")
        net = tf.nn.sigmoid(net)
        # net = conv_layer(x,    128, (5, 5), (2, 2), "SAME", "dc1")
        # net = conv_layer(net,  256, (5, 5), (2, 2), "SAME", "dc2")
        # net = conv_layer(net,  512, (5, 5), (2, 2), "SAME", "dc3")
        # net = conv_layer(net, 1024, (3, 3), (2, 2), "SAME", "dc4")
        return net


run_config = tf.ConfigProto()
run_config.allow_soft_placement = True
run_config.gpu_options.allow_growth = True
run_config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=run_config) as sess:
    print("creating dataset")
    data = Dataset({"dataset": "pokemon_sprites",
                    "batch_size": FLAGS.batch_size,
                    "height": FLAGS.input_height,
                    "width": FLAGS.input_width,
                    "channels": FLAGS.channels,
                    "train": FLAGS.train, })
    # "sess": sess})
    print("creating model")
    noise_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 100], name="noise")
    real_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 96, 96, 3], name="real")

    generated = generator(noise_placeholder)
    tf.summary.image("generated", generated)
    d_fake = discriminator(generated)
    d_real = discriminator(real_placeholder, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake))
    )
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake))
    )
    tf.summary.scalar("d_real", d_loss_real)
    tf.summary.scalar("d_fake", d_loss_fake)
    tf.summary.scalar("d_total", d_loss)
    tf.summary.scalar("g", g_loss)

    g_vars = [var for var in tf.trainable_variables() if "gen" in var.name]
    for var in g_vars:
        tf.summary.histogram("G_" + var.name, var)
    d_vars = [var for var in tf.trainable_variables() if "disc" in var.name]
    for var in d_vars:
        tf.summary.histogram("D_" + var.name, var)

    g_opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(g_loss, var_list=g_vars)
    d_opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(d_loss, var_list=d_vars)

    print("initializing variables")
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
    tf.global_variables_initializer().run()

    print("starting queue runners")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("starting training loop")
    show_all_variables()

    for i in range(FLAGS.iterations + 1):
        noise = np.random.normal(size=(FLAGS.batch_size, 100))
        ims = data.ims.eval()
        for j in range(FLAGS.g_train):
            _, g_loss_ = sess.run([g_opt, g_loss], {real_placeholder: ims,
                                                    noise_placeholder: noise})
        # val = sess.run(output, {placeholder: data.ims.eval()})
        _, d_loss_ = sess.run([d_opt, d_loss], {real_placeholder: ims,
                                                noise_placeholder: noise})

        print("Step {:<7d} - G loss {} - D loss {}".format(i, g_loss_, d_loss_))
        if i % 10 == 0:
            summary = sess.run(summary_op, {real_placeholder: ims,
                                            noise_placeholder: noise})
            summary_writer.add_summary(summary, i)

    coord.request_stop()
    coord.join(threads)
