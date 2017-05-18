from theano import function
from theano import pp
import theano.tensor as T

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2

print(pp(diff))
print(pp(abs_diff))
print(pp(diff_squared))

f = function([a, b], [diff, abs_diff, diff_squared])

d, abs, squared = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

print(d)
print(abs)
print(squared)