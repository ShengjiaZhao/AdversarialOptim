from abstract_network import *
Bernoulli = tf.contrib.distributions.Bernoulli


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def sample_z(batch_size, z_dim, model='gaussian'):
    if model == 'gaussian':
        return np.random.normal(0, 1, [batch_size, z_dim])
    elif model == 'bernoulli':
        return (np.random.normal(0, 1, [batch_size, z_dim]) > 0).astype(np.float)
    return None

def label_noise(bc):
    return (1.0 - np.abs(np.random.normal(loc=0, scale=0.1, size=bc.shape))) * bc + \
           np.abs(np.random.normal(loc=0, scale=0.1, size=bc.shape)) * (1 - bc)

# Encoder and decoder use the DC-GAN architecture
def encoder_discrete(x, z_dim, tau):
    with tf.variable_scope('i_net'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        z_logits = tf.reshape(tf.contrib.layers.fully_connected(fc1, z_dim * 2, activation_fn=tf.identity),
                              [-1, z_dim, 2])
        q_z = tf.nn.softmax(z_logits)
        # temperature
        z_sample = tf.reshape(gumbel_softmax(z_logits, tau, hard=True), [-1, z_dim, 2])[:, :, 0]
        return [q_z], z_sample


def encoder_gaussian(x, z_dim):
    with tf.variable_scope('i_net'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        sample = mean + tf.multiply(stddev, tf.random_normal(tf.stack([tf.shape(x)[0], z_dim])))
        return [mean, stddev], sample


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


def classifier(x, num_classes, reuse=False):
    with tf.variable_scope('c_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_classes, activation_fn=tf.identity)
        return fc2


def discriminator_cond(x, c, reuse=False):
    with tf.variable_scope('dc_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc = tf.concat([conv2, c], axis=1)
        fc = fc_bn_lrelu(fc, 1024)
        fc = tf.concat([fc, c], axis=1)
        fc = fc_bn_lrelu(fc, 1024)
        fc = tf.contrib.layers.fully_connected(fc, 1, activation_fn=tf.identity)
        return fc