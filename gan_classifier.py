from dataset import *
from abstract_network import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='mnist', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-6, help='learning rate')
parser.add_argument('-r', '--repeat', type=int, default=10, help='Number of times to train discriminator each time generator is trained')
args = parser.parse_args()

batch_size = 100
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = 'condgan/%s/-lr=%.2f-rep=%d' % (args.dataset, args.lr, args.repeat)
if args.dataset == 'mnist':
    dataset = MnistDataset(binary=False, one_hot=True)
else:
    dataset = CifarDataset(one_hot=True)

x = tf.placeholder(tf.float32, [None] + dataset.data_dims)
c = tf.placeholder(tf.float32, [None, 10])

c_ = classifier(x, 10)
d = discriminator_cond(x, c)
d_ = discriminator_cond(x, c_, reuse=True)

# Gradient penalty
epsilon = tf.random_uniform([], 0.0, 1.0)
c_hat = epsilon * c + (1 - epsilon) * c_
d_hat = discriminator_cond(x, c_hat, reuse=True)

ddc = tf.gradients(d_hat, c_hat)[0]
ddc = tf.sqrt(tf.reduce_sum(tf.square(ddc), axis=1))
d_grad_loss = tf.reduce_mean(tf.square(ddc - 1.0) * 10.0)

d_confusion = tf.reduce_mean(d_) - tf.reduce_mean(d)
d_loss = d_confusion + d_grad_loss
g_loss = -tf.reduce_mean(d_)

d_vars = [var for var in tf.global_variables() if 'dc_net' in var.name]
g_vars = [var for var in tf.global_variables() if 'c_net' in var.name]
d_train = tf.train.AdamOptimizer(learning_rate=10 ** args.lr, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_train = tf.train.AdamOptimizer(learning_rate=10 ** args.lr, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

correct_prediction = tf.equal(tf.argmax(c_, 1), tf.argmax(c, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_summary = tf.summary.merge([
    tf.summary.scalar('g_loss', g_loss),
    tf.summary.scalar('d_loss', d_loss),
    tf.summary.scalar('confusion', d_confusion),
    tf.summary.scalar('train_acc', accuracy),
])

test_summary = tf.summary.merge([
    tf.summary.scalar('test_acc', accuracy)
])

# sample_match_ph = tf.placeholder(tf.float32)
# sample_diversity_ph = tf.placeholder(tf.float32)
# sample_dist_ph = tf.placeholder(tf.float32, [None])
# sample_dist2_ph = tf.placeholder(tf.float32, [None])
# eval_summary = tf.summary.merge([
#     tf.summary.scalar('sample_match', sample_match_ph),
#     tf.summary.histogram('sample_dist', sample_dist_ph),
#     tf.summary.scalar('sample_diversity', sample_diversity_ph),
#     tf.summary.histogram('sample_diversity_dist', sample_dist2_ph),
#     create_display(tf.reshape(g, [100]+dataset.data_dims), 'samples'),
#     create_display(tf.reshape(x, [100]+dataset.data_dims), 'train_samples')
# ])

model_path = "log/%s" % name
make_model_path(model_path)
logger = open(os.path.join(model_path, 'result.txt'), 'w')
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
summary_writer = tf.summary.FileWriter(model_path)
sess.run(tf.global_variables_initializer())

start_time = time.time()
idx = 0
while True:
    bx, bc = dataset.next_labeled_batch(batch_size)
    bc = label_noise(bc)
    for i in range(args.repeat - 1):
        sess.run(d_train, feed_dict={x: bx, c: bc})
    sess.run([d_train, g_train], feed_dict={x: bx, c: bc})

    if idx % 100 == 0:
        summary_val, acc = sess.run([train_summary, accuracy], feed_dict={x: bx, c: bc})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f, accuracy=%.2f" % (idx, time.time() - start_time, acc))

    if idx % 500 == 0:
        bx, bc = dataset.next_labeled_test_batch(batch_size)
        bc = label_noise(bc)
        summary_val = sess.run(test_summary, feed_dict={x: bx, c: bc})
        summary_writer.add_summary(summary_val, idx)
    idx += 1