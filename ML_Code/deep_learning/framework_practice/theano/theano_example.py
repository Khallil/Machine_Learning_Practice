import theano
from theano import tensor

# declare variable types (floating-point scalars)
#scalar is quantity (numerical value)
a = tensor.dscalar()
b = tensor.dscalar()

# create a symbolic expression
c = a + b
# create the function with 1 : parameters, 2:symbolic expression
f = theano.function([a,b],c)

result = f(1.5,2.5)
print (result)