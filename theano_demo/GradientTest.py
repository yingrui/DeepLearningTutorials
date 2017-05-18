from theano import *
from unittest import *
import numpy
import theano.tensor as T

class GradientTest(TestCase):

    def testLinearFunctionGradForNumberInput(self):
        x, y = T.dscalars('x', 'y')
        z = x * x + 2 * y
        gx = T.grad(z, x)
        gy = T.grad(z, y)

        sx = shared(0., name="sx")
        sy = shared(0.)

        f = function(inputs=[x, y], outputs=z, updates=[(sx, gx), (sy, gy)])
        assert f(3, 3) == 15.
        assert 6. == sx.get_value()
        assert 2. == sy.get_value()
        print(sx.get_value(), sy.get_value())

    def testLinearFunctionGradForMatrixInput(self):
        x = T.matrix('x')
        y = T.vector('y')
        z = T.sum((2 * y) * x)

        gx, gy = T.grad(z, [x, y])

        sx = shared(numpy.asarray([[0, 0], [0, 0]], dtype="float64"), name="sx")
        sy = shared(numpy.asarray([0., 0.]))

        f = function(inputs=[x, y], outputs=z, updates=[(sx, gx), (sy, gy)])
        print(f(numpy.asarray([[1., 2.], [3., 4.]]), numpy.asarray([2., 3.])))
        print(sx.get_value(), sy.get_value())
