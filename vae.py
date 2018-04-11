from abstract_network import *
from dataset import *
import time

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
    with tf.variable_scope('encoder'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        z_logits = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        z_logits_pad = tf.pad(tf.expand_dims(z_logits, -1), paddings=[[0, 0], [0, 0], [0, 1]], mode='CONSTANT')
        z_sample = gumbel_softmax_sample(z_logits_pad, 1.0)[:, :, 0]
        return tf.nn.sigmoid(z_logits), z_sample


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        output = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        return output


z_dim = 12
dataset = RandomDataset(size=2 ** z_dim, one_hot=False)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
name = 'vae/%d' % (z_dim)

# Build the computation graph for training
batch_size = 100
train_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
train_zmean, train_zsample = encoder(train_x, z_dim)
train_xr = decoder(train_zsample)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_x = decoder(gen_z, reuse=True)


# ELBO loss divided by input dimensions
zkl_per_sample = tf.reduce_sum(train_zmean * tf.log(train_zmean + 1e-8) +
                               (1 - train_zmean) * tf.log(1 - train_zmean + 1e-8) + math.log(2.0), axis=1)
loss_zkl = tf.reduce_mean(zkl_per_sample)

# Negative log likelihood per dimension
nll_per_sample = tf.reduce_sum(tf.square(train_x - train_xr), axis=(1, 2, 3)) + math.log(2 * np.pi) / 2.0
loss_nll = tf.reduce_mean(nll_per_sample)
loss_elbo = loss_nll + loss_zkl
trainer = tf.train.AdamOptimizer(1e-3).minimize(loss_elbo)

train_summary = tf.summary.merge([
    tf.summary.scalar('loss_zkl', loss_zkl),
    tf.summary.scalar('loss_nll', loss_nll),
    tf.summary.scalar('loss_elbo', loss_elbo),
])

sample_match_ph = tf.placeholder(tf.float32)
sample_dist_ph = tf.placeholder(tf.float32, [None])
eval_summary = tf.summary.merge([
    tf.summary.scalar('sample_match', sample_match_ph),
    tf.summary.histogram('sample_dist', sample_dist_ph),
    create_display(tf.reshape(gen_x, [100]+dataset.data_dims), 'samples'),
    create_display(tf.reshape(train_xr, [100]+dataset.data_dims), 'reconstruction')
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

idx = 0
start_time = time.time()
while True:
    bx = dataset.next_batch(batch_size)
    _, nll, zkl = sess.run([trainer, loss_elbo, loss_zkl], feed_dict={train_x: bx})

    if idx % 10 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: bx})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 100 == 0:
        bz = (np.random.normal(0, 1, [batch_size, z_dim]) > 0).astype(np.float)
        samples = sess.run(gen_x, feed_dict={train_x: bx, gen_z: bz})
        sample_match, sample_dist = dataset.compare(samples)
        summary_val = sess.run(eval_summary,
                               feed_dict={sample_match_ph: sample_match, sample_dist_ph: sample_dist, train_x: bx, gen_z: bz})
        summary_writer.add_summary(summary_val, idx)
    idx += 1