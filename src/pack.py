import numpy as np
import joblib


def weight(x, y):
    return 1. / max(1e-6, np.sum(np.square(x - y)))


class Pack(object):
    def __init__(self, key_points):
        self.key_points = key_points
        self.keys, self.values = zip(*key_points)
        self.n = len(self.key_points)

    def get(self, point):
        s = 0.
        sw = 0.
        for i in range(self.n):
            w = weight(point, self.keys[i])
            s += w * self.values[i]
            sw += w
        return s / sw

    @staticmethod
    def from_file(filename):
        key_points = joblib.load(filename)
        return Pack(key_points)
