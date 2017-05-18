from theano import *
from unittest import *

class MatrixTest(TestCase):

    def test_CreateMatrix(self):
        m = numpy.asarray([[1., 2], [3, 4], [5, 6]])
        print(m[0, 0], m[2, 1])
        assert (m[0, 0] == 1.0) & (m[2, 1] == 6.0)
        assert m.shape == (3, 2)

    def test_MatrixMultiple(self):
        m = numpy.asarray([[1., 2], [3, 4], [5, 6]])
        m *= 2
        assert (m[0, 0] == 2.0) & (m[2, 1] == 12.0)