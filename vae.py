from abstract_network import *
from dataset import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-z', '--z_dim', type=int, default=7, help='z dimension')
parser.add_argument('-a', '--z_add', type=int, default=20, help='Additional dimensions to add to z')
parser.add_argument('-m', '--model', type=str, default='gaussian', help='gaussian or discrete')
parser.add_argument('-d', '--dataset', type=str, default='random', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-4.0, help='learning rate')
parser.add_argument('--beta', type=float, default=1.0, help='Coefficient of KL(q(z|x)||p(z))')
args = parser.parse_args()

# Hypothesis: optimization gets stuck in local minimum and do not differentiate between the different x
z_dim = args.z_dim
train_size = 2 ** z_dim
batch_size = 100
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = 'vae/%s/model=%s-zdim=%d-zadd=%d-lr=%.2f-beta=%.2f' % (args.dataset, args.model, z_dim, args.z_add, args.lr, args.beta)
z_dim += args.z_add
if args.dataset == 'mnist':
    dataset = MnistDataset(binary=False)
else:
    dataset = RandomDataset(size=train_size, binary=False)

# Build the computation graph for training
train_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
tau = tf.placeholder(tf.float32, shape=[], name="temperature")
if args.model == 'gaussian':
    train_zdist, train_zsample = encoder_gaussian(train_x, z_dim)
    # ELBO loss divided by input dimensions
    zkl_per_sample = tf.reduce_sum(-tf.log(train_zdist[1]) + 0.5 * tf.square(train_zdist[1]) +
                                   0.5 * tf.square(train_zdist[0]) - 0.5, axis=1)
    loss_zkl = tf.reduce_mean(zkl_per_sample)
else:
    assert args.model == 'bernoulli'
    train_zdist, train_zsample = encoder_discrete(train_x, z_dim, tau)
    q_z = train_zdist[0]
    log_q_z = tf.log(train_zdist[0] + 1e-20)
    zkl_per_sample = tf.reduce_sum(q_z * (log_q_z - tf.log(1.0 / 2.0)), [1, 2])
    loss_zkl = tf.reduce_mean(zkl_per_sample) * 0.9
train_xr = generator(train_zsample)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_x = generator(gen_z, reuse=True)

# Negative log likelihood per dimension
nll_per_sample = 64 * tf.reduce_sum(tf.square(train_x - train_xr) + 0.5 * tf.abs(train_x - train_xr), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(nll_per_sample)

loss_elbo = loss_nll + loss_zkl
trainer = tf.train.AdamOptimizer(10 ** args.lr, beta1=0.5, beta2=0.9).minimize(loss_elbo)

train_summary = tf.summary.merge([
    tf.summary.scalar('loss_zkl', loss_zkl),
    tf.summary.scalar('loss_nll', loss_nll),
    tf.summary.scalar('loss_elbo', loss_elbo),
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

idx = 0
start_time = time.time()
tau0 = 1.0 # initial temperature
np_temp = tau0
while True:
    bx = dataset.next_batch(batch_size)
    _, nll, zkl = sess.run([trainer, loss_elbo, loss_zkl], feed_dict={train_x: bx, tau: np_temp})

    if idx % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: bx, tau: np_temp})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f" % (idx, time.time() - start_time))

    if idx % 500 == 0:
        sample_list = []
        sample_cnt = 0
        while sample_cnt < train_size:
            bz = sample_z(min(train_size-sample_cnt, 100), z_dim, args.model)
            samples = sess.run(train_xr, feed_dict={train_x: bx, gen_z: bz})
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