from dataset import *
from abstract_network import *
import time


def generator(z, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
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


z_dim = 12
batch_size = 100
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
z_dist = 'gaussian'
name = 'gan/%s_%d' % (z_dist, z_dim)
dataset = RandomDataset(size=2 ** z_dim)


def sample_z(batch_size):
    if z_dist == 'gaussian':
        return np.random.normal(0, 1, [batch_size, z_dim])
    elif z_dist == 'bernoulli':
        return (np.random.normal(0, 1, [batch_size, z_dim]) > 0).astype(np.float)
    return None


z = tf.placeholder(tf.float32, [None, z_dim])
x = tf.placeholder(tf.float32, [None] + dataset.data_dims)

g = generator(z)
d = discriminator(x)
d_ = discriminator(g, reuse=True)

# Gradient penalty
epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = epsilon * x + (1 - epsilon) * g
d_hat = discriminator(x_hat, reuse=True)

ddx = tf.gradients(d_hat, x_hat)[0]
ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

d_loss_x = -tf.reduce_mean(d)
d_loss_g = tf.reduce_mean(d_)
d_loss = d_loss_x + d_loss_g + d_grad_loss
d_confusion = tf.reduce_mean(d) - tf.reduce_mean(d_)
g_loss = -tf.reduce_mean(d_)

d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]
d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

train_summary = tf.summary.merge([
    tf.summary.scalar('g_loss', g_loss),
    tf.summary.scalar('d_loss', d_loss),
    tf.summary.scalar('confusion', d_confusion),
    tf.summary.scalar('d_loss_g', d_loss_g),
])

sample_match_ph = tf.placeholder(tf.float32)
sample_dist_ph = tf.placeholder(tf.float32, [None])
eval_summary = tf.summary.merge([
    tf.summary.scalar('sample_match', sample_match_ph),
    tf.summary.histogram('sample_dist', sample_dist_ph),
    create_display(tf.reshape(g, [100]+dataset.data_dims), 'samples'),
    create_display(tf.reshape(x, [100]+dataset.data_dims), 'train_samples')
])


def make_model_path(model_path):
    import subprocess
    if os.path.isdir(model_path):
        subprocess.call(('rm -rf %s' % model_path).split())
    os.makedirs(model_path)


model_path = "log/%s" % name
make_model_path(model_path)
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
summary_writer = tf.summary.FileWriter(model_path)
sess.run(tf.global_variables_initializer())

start_time = time.time()
idx = 0
while True:
    bx = dataset.next_batch(batch_size)
    bz = sample_z(batch_size)
    sess.run([d_train, g_train], feed_dict={x: bx, z: bz})

    if idx % 10 == 0:
        summary_val = sess.run(train_summary, feed_dict={x: bx, z: bz})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 100 == 0:
        bz = sample_z(100)
        samples = sess.run(g, feed_dict={z: bz})
        sample_match, sample_dist = dataset.compare(samples)
        summary_val = sess.run(eval_summary,
                               feed_dict={sample_match_ph: sample_match, sample_dist_ph: sample_dist, x: bx, z: bz})
        summary_writer.add_summary(summary_val, idx)
    idx += 1