from dataset import *
from abstract_network import *
import time
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='mnist', help='mnist or random')
parser.add_argument('-lr', '--lr', type=float, default=-4, help='learning rate')
parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of times to train discriminator each time generator is trained')
args = parser.parse_args()

batch_size = 100
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = 'condsupervised/%s/-lr=%.2f-rep=%d' % (args.dataset, args.lr, args.repeat)
if args.dataset == 'mnist':
    dataset = MnistDataset(binary=False, one_hot=True)
else:
    dataset = CifarDataset(one_hot=True)

x = tf.placeholder(tf.float32, [None] + dataset.data_dims)
c = tf.placeholder(tf.float32, [None, 10])

c_ = classifier(x, 10)
c_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(c - c_), axis=1))
c_train = tf.train.AdamOptimizer(learning_rate=10 ** args.lr, beta1=0.5, beta2=0.9).minimize(c_loss)

correct_prediction = tf.equal(tf.argmax(c_, 1), tf.argmax(c, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_summary = tf.summary.merge([
    tf.summary.scalar('c_loss', c_loss),
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
    sess.run(c_train, feed_dict={x: bx, c: bc})

    if idx % 100 == 0:
        summary_val, acc = sess.run([train_summary, accuracy], feed_dict={x: bx, c: bc})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration: [%6d] time: %4.2f, accuracy=%.2f" % (idx, time.time() - start_time, acc))

    if idx % 500 == 0:
        bx, bc = dataset.next_labeled_test_batch(batch_size)
        bc = label_noise(bc)
        summary_val = sess.run(test_summary, feed_dict={x: bx, c: bc})
        summary_writer.add_summary(summary_val, idx)
    # if idx % 500 == 0:
    #     sample_list = []
    #     sample_cnt = 0
    #     while sample_cnt < train_size:
    #         bz = sample_z(min(train_size-sample_cnt, 100), z_dim, args.model)
    #         samples = sess.run(g, feed_dict={z: bz})
    #         sample_cnt += 100
    #         sample_list.append(samples)
    #     samples = np.concatenate(sample_list, axis=0)
    #     sample_match, sample_dist = dataset.compare(samples)
    #     sample_diversity, sample_dist2 = dataset.diversity(samples)
    #     logger.write('%d %.3f %.3f\n' % (idx, sample_match, sample_diversity))
    #     logger.flush()
    #     summary_val = sess.run(eval_summary,
    #                            feed_dict={sample_match_ph: sample_match, sample_dist_ph: sample_dist,
    #                                       sample_diversity_ph: sample_diversity, sample_dist2_ph: sample_dist2,
    #                                       x: bx, z: sample_z(100, z_dim, args.model)})
    #     summary_writer.add_summary(summary_val, idx)
    idx += 1