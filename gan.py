from dataset import *
from abstract_network import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=10, help='z dimension')
parser.add_argument('-a', '--z_add', type=int, default=10, help='Additional dimensions to add to z')
parser.add_argument('-m', '--model', type=str, default='bernoulli', help='gaussian or discrete')
parser.add_argument('-d', '--dataset', type=str, default='random', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-4, help='learning rate')
parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of times to train discriminator each time generator is trained')
args = parser.parse_args()

z_dim = args.z_dim
train_size = 2 ** z_dim
batch_size = 100
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = 'gan/%s/model=%s-zdim=%d-zadd=%d-lr=%.2f-rep=%d' % (args.dataset, args.model, z_dim, args.z_add, args.lr, args.repeat)
z_dim += args.z_add
if args.dataset == 'mnist':
    dataset = MnistDataset(binary=False)
else:
    dataset = RandomDataset(size=train_size, binary=False)

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
d_train = tf.train.AdamOptimizer(learning_rate=10 ** args.lr, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_train = tf.train.AdamOptimizer(learning_rate=10 ** args.lr, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

train_summary = tf.summary.merge([
    tf.summary.scalar('g_loss', g_loss),
    tf.summary.scalar('d_loss', d_loss),
    tf.summary.scalar('confusion', d_confusion),
    tf.summary.scalar('d_loss_g', d_loss_g),
])

sample_match_ph = tf.placeholder(tf.float32)
sample_diversity_ph = tf.placeholder(tf.float32)
sample_dist_ph = tf.placeholder(tf.float32, [None])
sample_dist2_ph = tf.placeholder(tf.float32, [None])
eval_summary = tf.summary.merge([
    tf.summary.scalar('sample_match', sample_match_ph),
    tf.summary.histogram('sample_dist', sample_dist_ph),
    tf.summary.scalar('sample_diversity', sample_diversity_ph),
    tf.summary.histogram('sample_diversity_dist', sample_dist2_ph),
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
    bx = dataset.next_batch(batch_size)
    bz = sample_z(batch_size, z_dim, args.model)
    for i in range(args.repeat - 1):
        sess.run(d_train, feed_dict={x: bx, z: bz})
    sess.run([d_train, g_train], feed_dict={x: bx, z: bz})

    if idx % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={x: bx, z: bz})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 500 == 0:
        sample_list = []
        sample_cnt = 0
        while sample_cnt < train_size:
            bz = sample_z(min(train_size-sample_cnt, 100), z_dim, args.model)
            samples = sess.run(g, feed_dict={z: bz})
            sample_cnt += 100
            sample_list.append(samples)
        samples = np.concatenate(sample_list, axis=0)
        sample_match, sample_dist = dataset.compare(samples)
        sample_diversity, sample_dist2 = dataset.diversity(samples)
        logger.write('%d %.3f %.3f\n' % (idx, sample_match, sample_diversity))
        logger.flush()
        summary_val = sess.run(eval_summary,
                               feed_dict={sample_match_ph: sample_match, sample_dist_ph: sample_dist,
                                          sample_diversity_ph: sample_diversity, sample_dist2_ph: sample_dist2,
                                          x: bx, z: sample_z(100, z_dim, args.model)})
        summary_writer.add_summary(summary_val, idx)
    idx += 1