from theano import *
from unittest import *

import theano.tensor as T

class LogisticFunctionTest(TestCase):

    def test_SigmoidFunction(self):
        x = T.dmatrix('x')
        sigmoid = 1 / (1 + T.exp(-x))
        logistic = function([x], sigmoid)
        y = logistic([[0, 1], [-1, -2]])
        assert y.shape == (2, 2)
        print y

    def test_TanhFunction(self):
        x = T.dmatrix('x')
        tanh = (1 + T.tanh(x / 2)) / 2
        logistic = function([x], tanh)
        y = logistic([[0, 1], [-1, -2]])
        assert y.shape == (2, 2)
        print y

