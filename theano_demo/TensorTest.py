from unittest import *
from theano import *
from theano import pp
from theano import Param
from theano import shared
import theano.tensor as T

class TensorTest(TestCase):

    def test_CreateFunction(self):
        x = T.dscalar('x')
        y = T.dscalar('y')
        z = x + y
        print(pp(z))

        f = function([x, y], z)
        print(f(2, 3))
        assert f(2, 3) == 2 + 3

    def test_AddMatrix(self):
        m = T.dmatrix('m')
        n = T.dmatrix('n')

        y = m + n
        f = function([m, n], y)

        result = f([[1, 2], [3, 4]], [[4, 3], [2, 1]])
        assert result.shape == (2, 2)

    def test_ComputeSigmoidAndTanhAtSameTime(self):
        x = T.dmatrix('x')
        sigmoid = 1 / (1 + T.exp(-x))
        tanh = (1 + T.tanh(x / 2)) / 2
        logistic = function([x], [sigmoid, tanh])
        y = logistic([[0, 1], [-1, -2]])
        assert len(y) == 2
        print(y[0], y[1])

    def test_SetDefaultValueForAnArgument(self):
        x, y = T.dscalars('x', 'y')
        z = x + y
        f = function([x, Param(y, default=1)], z)
        result = f(33)
        assert result == 34
        result = f(33, 2)
        assert result == 35

    def test_SetByNameDefaultArgument(self):
        x, y, w = T.dscalars('x', 'y', 'w')
        z = (x + y) * w
        f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
        result = f(33)
        assert result == 68
        result = f(33, 2)
        assert result == 70
        result = f(33, w_by_name=1)
        assert result == 34
        result = f(33, w_by_name=1, y=0)
        assert result == 33

    def test_SharedValue(self):
        state = shared(0)
        inc = T.iscalar('inc')
        accmulator = function([inc], state, updates=[(state, state+inc)])
        assert state.get_value() == 0
        accmulator(1)
        assert state.get_value() == 1

        state.set_value(0)
        assert state.get_value() == 0
