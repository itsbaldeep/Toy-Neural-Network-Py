from random import randint
from Network import Network

brain = Network(2, 5, 1)
xs = [ [1,0], [0,1], [1,1], [0,0] ]
ys = [ [1], [1], [0], [0] ]

for _ in range(10000):
    ind = randint(0, len(xs) - 1)
    brain.train(xs[ind], ys[ind])

for x in xs:
    print(brain.predict(x)[0])
