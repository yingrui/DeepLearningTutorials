import numpy
from theano import function
from theano import pp
import theano.tensor as T

x = T.dscalar('x')
y = T.dscalar('y')

z = x + y

print(pp(z))

f = function([x, y], z)

print(numpy.allclose(f(16.3, 12.1), 28.4))
print(numpy.allclose(z.eval({x: 16.3, y: 12.1}), 28.4))
