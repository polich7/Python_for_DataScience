#Обучить нейрон, представленный в видео https://www.youtube.com/watch?v=SEukWq_e3Hs
import numpy as np
w = np.zeros((3))

D = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
])

Y0 = np.array([0, 1, 0, 0, 1])

a = 0.2
b = -0.4
c = lambda x: 1 if x > 0 else 0


def f(x):
    s = b + np.sum(x @ w)
    return c(s)


def train():
    global w
    _w = w.copy()
    for x, y in zip(D, Y0):
        w += a * (y - f(x)) * x
    return (w != _w).any()


while train():
    print(w)

print(f([1, 1, 1]))
print(f([1, 1, 0]))