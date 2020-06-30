#Обучить нейрон, представленный в видео https://www.youtube.com/watch?v=SEukWq_e3Hs распознавать буквы.
import numpy as np
import matplotlib.pyplot as plt
import os

w = np.zeros((25))  #веса
D = None
Y0 = np.array([1,0,0,1,0,0,1,0,0], dtype=float)  #у для обучающей выборки

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
# выгружаем все картинки из папки для обучения
path_train='2_2_train/'
for name in os.listdir(path_train):
    img = plt.imread('{}{}'.format(path_train, name))
    #print(img)
    xs = np.dot(img[...,:3], [1, 1, 1]) .flatten()  #Привели к ч/б умножив срез на [1, 1, 1]+ превращаем матрицу в вектор
    if D is None:
        D = xs
    else:
        D = np.vstack((D, xs))

#print(D)
print(Y0)

while train():
    print(w)

path_test = '2_2_test/'
for name in os.listdir(path_test):
    img = plt.imread('{}{}'.format(path_test, name))
    #print(img)
    xs = np.dot(img[..., :3], [1, 1, 1]) .flatten()  #Привели к ч/б умножив срез на [1, 1, 1]+ превращаем матрицу в вектор
    print(f(xs))