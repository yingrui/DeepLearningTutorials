from unittest import *
from theano import *
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

class Graph():
    def __init__(self, seed=123):
        self.rng = RandomStreams(seed)
        self.y = self.rng.uniform(size=(1,))

class MRGGraph():
    def __init__(self, seed=123):
        self.rng = MRG_RandomStreams(seed)
        self.y = self.rng.uniform(size=(1,))

class RandomStreamTest(TestCase):

    def test_RandomStream(self):
        srng = RandomStreams(seed=234)
        rv_u = srng.uniform((2, 2))
        rv_n = srng.normal((2, 2))
        f = function([], rv_u)
        g = function([], rv_n, no_default_updates=True)
        nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

        print(f(), f())

        print(g(), g())

        print(nearly_zeros(), nearly_zeros())

    def test_copyRandomState(self):
        g1 = Graph(123)
        f1 = function([], g1.y)

        g2 = Graph(seed=987)
        f2 = function([], g2.y)

        print('By default, the two functions are out of sync.')
        print('f1() returns ', f1())
        print('f2() returns ', f2())
        def copy_random_state(g1, g2):
            for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
                su2[0].set_value(su1[0].get_value())

        print('We now copy the state of the theano random number generators.')
        copy_random_state(g1, g2)
        print('f1() returns ', f1())
        print('f2() returns ', f2())

    def test_copyMRGRandomState(self):
        g1 = MRGGraph(123)
        f1 = function([], g1.y)

        g2 = MRGGraph(seed=987)
        f2 = function([], g2.y)

        print('By default, the two functions are out of sync.')
        print('f1() returns ', f1())
        print('f2() returns ', f2())

        def copy_random_state(g1, g2):
            if isinstance(g1.rng, MRG_RandomStreams):
                g2.rng.rstate = g1.rng.rstate

        print('We now copy the state of the theano random number generators.')
        copy_random_state(g1, g2)
        print('f1() returns ', f1())
        print('f2() returns ', f2())
