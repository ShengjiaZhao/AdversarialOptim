from abstract_network import *
from dataset import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=12, help='z dimension')
args = parser.parse_args()

# Hypothesis: optimization gets stuck in local minimum and do not differentiate between the different x
z_dim = 7
additional_z = 10
dataset = RandomDataset(size=2 ** z_dim, one_hot=False)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
name = 'vae/%d-%d' % (z_dim, additional_z)
z_dim += additional_z

# Build the computation graph for training
batch_size = 100
train_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
train_zmean, train_zsample = encoder_gaussian(train_x, z_dim)
train_xr = generator(train_zsample)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_x = generator(gen_z, reuse=True)


# ELBO loss divided by input dimensions
zkl_per_sample = tf.reduce_sum(train_zmean * tf.log(train_zmean + 1e-8) +
                               (1 - train_zmean) * tf.log(1 - train_zmean + 1e-8) + math.log(2.0), axis=1)
loss_zkl = tf.reduce_mean(zkl_per_sample) * 0.00

# Negative log likelihood per dimension
nll_per_sample = tf.reduce_sum(tf.abs(train_x - train_xr), axis=(1, 2, 3))
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