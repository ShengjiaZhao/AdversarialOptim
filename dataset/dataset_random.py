if __name__ == '__main__':
    from abstract_dataset import *
else:
    from .abstract_dataset import *
from sklearn.neighbors import NearestNeighbors


class RandomDataset(Dataset):
    def __init__(self, size=10000, dim=(28, 28, 1), one_hot=False, binary=True):
        Dataset.__init__(self)
        self.name = "mnist"
        self.data_dims = list(dim)
        self.train_size = size
        self.test_size = size
        self.range = [0.0, 1.0]
        self.train_data = np.random.normal(loc=0, scale=1.0, size=[size] + self.data_dims)
        self.train_data = 1 / (1 + np.exp(-self.train_data))
        if binary:
            self.train_data = (self.train_data > 0.5).astype(np.int)
        if one_hot:
            self.labels = np.arange(0, self.train_size)
        else:
            bits = int(math.ceil(math.log(size, 2)))
            self.labels = np.array([[int(j) for j in ('{0:0%db}' % bits).format(i)] for i in range(self.train_size)])
        self.test_data = self.train_data
        self.train_ptr = 0
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
            np.reshape(self.train_data, newshape=[-1, np.prod(dim)]))

    def next_batch(self, batch_size):
        return self.next_labeled_batch(batch_size)[0]

    def next_labeled_batch(self, batch_size):
        assert batch_size <= self.train_size
        prev_ptr = self.train_ptr
        self.train_ptr += batch_size
        if self.train_ptr > self.train_size:
            self.train_ptr -= self.train_size
            img = np.concatenate([self.train_data[prev_ptr:],
                                  self.train_data[:self.train_ptr]], axis=0)
            label = np.concatenate([self.labels[prev_ptr:],
                                   self.labels[:self.train_ptr]])
        else:
            img = self.train_data[prev_ptr:self.train_ptr]
            label = self.labels[prev_ptr:self.train_ptr]
        return img, label

    def next_test_batch(self, batch_size):
        return self.next_batch(batch_size)

    def next_labeled_test_batch(self, batch_size):
        return self.next_labeled_batch(batch_size)

    def display(self, image):
        return np.clip(image, a_min=0.0, a_max=1.0)

    def compare(self, images):
        distances, indices = self.nbrs.kneighbors(np.reshape(images, newshape=[-1, np.prod(self.data_dims)]))
        return np.mean(distances), distances.flatten()

    def diversity(self, images):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(
            np.reshape(images, newshape=[-1, np.prod(self.data_dims)])
        )
        distances, indices = nbrs.kneighbors(np.reshape(images, newshape=[-1, np.prod(self.data_dims)]))
        return np.mean(distances[:, 1]), distances[:, 1].flatten()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import time
    one_hot = False
    dataset = RandomDataset(size=5000, one_hot=one_hot, binary=True)
    for i in range(2):
        sample, label = dataset.next_labeled_batch(30)
        for index in range(30):
            plt.subplot(6, 10, index+1+30*i)
            plt.imshow(sample[index, :, :, 0].astype(np.float), vmin=0.0, vmax=1.0, cmap=plt.get_cmap('Greys'))
            if not one_hot:
                plt.title(''.join(['%d' % l for l in label[index]]))
            else:
                plt.title('%d' % label[index])
        print(dataset.compare(sample)[0])
    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()
    plt.show()

    start_time = time.time()
    distance, dist = dataset.diversity(dataset.train_data)
    elapsed_time = time.time() - start_time
    print("Time used %.2f, distance=%.2f" % (elapsed_time, distance))
    plt.hist(dist, bins=20)
    plt.show()