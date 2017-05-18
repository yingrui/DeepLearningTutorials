from theano import function
from theano import shared
import theano.tensor as T

state = shared(0)

inc = T.iscalar('inc')

accumulator = function([inc], state, updates=[(state, state+inc)])

print(accumulator(1))
print(state.get_value())

'''
You might be wondering why the updates mechanism exists.
You can always achieve a similar result by returning the new expressions, and working with them in NumPy as usual.
The updates mechanism can be a syntactic convenience, but it is mainly there for efficiency.
Updates to shared variables can sometimes be done more quickly using in-place algorithms (e.g. low-rank matrix updates).
Also, Theano has more control over where and how shared variables are allocated,
which is one of the important elements of getting good performance on the GPU.
'''


'''
It may happen that you expressed some formula using a shared variable, but you do not want to use its value.
In this case, you can use the givens parameter of function which replaces a particular node in a graph for
the purpose of one particular function.
'''
fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print(skip_shared(1, 3))
print(state.get_value())

'''
The givens parameter can be used to replace any symbolic variable, not just a shared variable.

In practice, a good way of thinking about the givens is as a mechanism that allows you to
replace any part of your formula with a different expression that evaluates to a tensor of same shape and dtype.
'''