from abstract_network import *


def sample_gumbel(shape, eps=1e-20):
    u = tf.random_uniform(shape, minval=0.0, maxval=1.0)
    return -tf.log(-tf.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(y / temperature)

    # Make hard samples
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keep_dims=True)), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
    return y


# Encoder and decoder use the DC-GAN architecture
def encoder(x, z_dim):
    with tf.variable_scope('i_net'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        z_logits = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        z_logits_pad = tf.pad(tf.expand_dims(z_logits, -1), paddings=[[0, 0], [0, 0], [0, 1]], mode='CONSTANT')
        z_sample = gumbel_softmax_sample(z_logits_pad, 1.0)[:, :, 0]
        return tf.nn.sigmoid(z_logits), z_sample


def encoder_gaussian(x, z_dim):
    with tf.variable_scope('i_net'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        z_logits = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        return z_logits, z_logits


def generator(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc = fc_relu(z, 1024)
        fc = fc_relu(fc, 7*7*128)
        fc = tf.reshape(fc, tf.stack([tf.shape(fc)[0], 7, 7, 128]))
        conv = conv2d_t_relu(fc, 64, 4, 2)
        conv = conv2d_t_relu(conv, 64, 4, 1)
        output = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.sigmoid)
        return output


def discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return fc2