from dataset import *
from abstract_network import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=12, help='z dimension')
args = parser.parse_args()

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