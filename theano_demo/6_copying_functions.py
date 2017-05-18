import theano
import theano.tensor as T

state = theano.shared(0)
inc = T.iscalar('inc')

accumulator = theano.function([inc], state, updates=[(state, state + inc)])

accumulator(10)

print(state.get_value())

new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state: new_state})

new_accumulator(100)
print(state.get_value())
print(new_state.get_value())
