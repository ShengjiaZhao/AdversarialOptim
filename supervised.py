from dataset import *
from abstract_network import *
import time
import argparse
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=12, help='z dimension')
args = parser.parse_args()


z_dim = args.z_dim
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 100
name = 'supervised_gen/%d' % z_dim
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

model_path = "log/%s" % name
make_model_path(model_path)
logger = open(os.path.join(model_path, 'result.txt'), 'w')
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
        logger.write('%d %.3f\n' % (idx, sample_match))
        logger.flush()
        summary_val = sess.run(eval_summary,
                               feed_dict={sample_match_ph: sample_match, sample_dist_ph: sample_dist, x: bx, z: bz})
        summary_writer.add_summary(summary_val, idx)
    idx += 1
