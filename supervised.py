from dataset import *
from abstract_network import *
import time


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


z_dim = 12
batch_size = 100
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
name = 'supervised_gen2/%d' % z_dim
dataset = RandomDataset(size=2 ** z_dim, one_hot=False)

z = tf.placeholder(tf.float32, [None, z_dim])
x = tf.placeholder(tf.float32, [None] + dataset.data_dims)

g = generator(z)
supervised_loss = tf.reduce_mean(tf.abs(x - g))
train = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.5, beta2=0.9).minimize(supervised_loss)

train_summary = tf.summary.merge([
    tf.summary.scalar('supervised_loss', supervised_loss),
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
    bx, bz = dataset.next_labeled_batch(batch_size)
    sess.run(train, feed_dict={x: bx, z: bz})

    if idx % 10 == 0:
        summary_val = sess.run(train_summary, feed_dict={x: bx, z: bz})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 100 == 0:
        bz = (np.random.normal(0, 1, [batch_size, z_dim]) > 0).astype(np.float)
        samples = sess.run(g, feed_dict={z: bz})
        sample_match, sample_dist = dataset.compare(samples)
        summary_val = sess.run(eval_summary,
                               feed_dict={sample_match_ph: sample_match, sample_dist_ph: sample_dist, x: bx, z: bz})
        summary_writer.add_summary(summary_val, idx)
    idx += 1