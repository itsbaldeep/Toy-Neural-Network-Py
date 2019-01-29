# Toy Neural Network in Python
This is a machine learning library made in Python. This is a very basic and simple library intended towards students who are beginners in machine learning. The whole library is influenced by Daniel Shiffman's Toy Neural Network JS Library

# Usage
```python
# Initializing
brain = Network(2, 3, 4)

# Training
for _ in range(10000):
  brain.train(xs, ys)

# Predicting
print(brain.predict(xs))
```

# Motivation
This python program is made possible by [Daniel Shiffman's Neural Network Playlist.](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)

Also, I have already re-done the library [in Java](https://github.com/itsbaldeep/Toy-Neural-Network-Java/) Have a look there as well.

As compared to my Java version of this (which has ~450 lines of code), this is about 88.8% smaller (with ~50 lines of code) due to numpy and the nature of Python itself.

# Description
1. It supports only a single hidden layer and an output layer.
2. Can have as many number of hidden nodes.
3. It uses numpy as a dependency, so the program is efficient and compact.

# Dependencies
Make sure you have Numpy installed.
```
pip install numpy
```

# Running this repo
Network.py file contains just the implementation of the Neural Network. So open XOR.py to actually run the program.
```
python XOR.py
```
