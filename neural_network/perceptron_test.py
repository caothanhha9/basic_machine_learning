from random import choice
from numpy import array, dot, random


unit_step = lambda x_: 0 if x_ < 0 else 1
training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

w = random.rand(3, 1)
w = w.reshape(3)
errors = []
eta = 0.2
n = 1000
deg_eta = 0.99
factor = eta
for i in xrange(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += factor * error * x
    factor *= deg_eta


for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x, result, unit_step(result)))


