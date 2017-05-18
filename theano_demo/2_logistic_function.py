import numpy
from theano import function
from theano import pp
import theano.tensor as T

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))

print(pp(s))

logistic = function([x], s)

i = [[0, 1], [-1, -2]]

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)

print(logistic(i))
print(numpy.allclose(logistic(i), logistic2(i)))
