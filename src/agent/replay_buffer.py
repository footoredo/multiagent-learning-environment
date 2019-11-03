import random


class ReplayBuffer(object):
    def __init__(self, n_items, size):
        self.n_items = n_items
        self.size = size
        self._true_size = 0
        self._total = 0
        self._storage = []

    def add(self, items):
        self._total += 1
        if self._true_size < self.size:
            self._storage.append(items)
            self._true_size += 1
        else:
            if random.random() < self.size / self._total:
                self._storage[random.randrange(self.size)] = items

    def sample(self, sample_size):
        assert self._true_size >= sample_size
        sample_index = random.sample(range(self._true_size), sample_size)
        sample = [[None for _ in range(sample_size)] for _ in range(self.n_items)]
        for i, idx in enumerate(sample_index):
            for ii in range(self.n_items):
                sample[ii][i] = self._storage[idx][ii]
        return sample
