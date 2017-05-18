from theano import function
from theano import pp
from theano import In
import theano.tensor as T

x, y = T.dscalars('x', 'y')
z = x + y

print(pp(z))

f = function([x, In(y, value=1)], z)
print(f(33))
print(f(33, 2))