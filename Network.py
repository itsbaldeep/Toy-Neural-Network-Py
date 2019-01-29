import numpy as np
from numpy.random import random

class Network():

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self._in = input_nodes
        self._hi = hidden_nodes
        self._ou = output_nodes

        self.weights_ih = random([self._hi, self._in])
        self.weights_ho = random([self._ou, self._hi])
        self.bias_h = random([self._hi, 1])
        self.bias_o = random([self._ou, 1])

        self.lr = .2
        self.af = lambda x: 1 / (1 + np.exp(-x))
        self.daf = lambda x: x * (1 - x)


    def predict(self, xs):
        xs = np.array(xs).reshape(self._in, 1)
        hid = self.af(self.weights_ih @ xs + self.bias_h)
        out = self.af(self.weights_ho @ hid + self.bias_o)
        return out.flatten()

    def train(self, xs, ys):
        xs = np.array(xs).reshape(self._in, 1)
        ys = np.array(ys).reshape(self._ou, 1)
        hid = self.af(self.weights_ih @ xs + self.bias_h)
        out = self.af(self.weights_ho @ hid + self.bias_o)

        out_err = ys - out
        out_grad = self.daf(out) * out_err * self.lr
        self.weights_ho += out_grad @ hid.T
        self.bias_o += out_grad

        hid_err = self.weights_ho.T @ out_err
        hid_grad = self.daf(hid) * hid_err * self.lr
        self.weights_ih += hid_grad @ xs.T
        self.bias_h += hid_grad

