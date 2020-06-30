#На основе нейрона из видео https://www.youtube.com/watch?v=SEukWq_e3Hs сделать однослойный перцептрон
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
D = None
Y=None
w = np.zeros((5, 25))
a = 0.2
b = -0.4
c = lambda x: 1 if x > 0 else 0

def f(x,i):
    s = b + np.sum(x @ w[i])
    return c(s)

def train(i):
    global w
    _w = w[i].copy()
    for x, y in zip(D, Y[i]):
        w[i] += a * (y - f(x,i)) * x
    return (w[i] != _w).any()
def net(xs):
    s = ''
    for i in range(np.shape(w)[0]):
        s = '{}{}'.format(s, f(xs, i))
    return(s)

# выгружаем все картинки из папки для обучения
path_train='2_2_train/'
for name in os.listdir(path_train):
    img = plt.imread('{}{}'.format(path_train, name))
    #print(img)

    xs = (np.dot(img[...,:3], [1, 1, 1])).astype(int).flatten()  #Привели к ч/б умножив срез на [1, 1, 1]+ превращаем матрицу в вектор
    #print(xs)
    #print(type(xs))
    ys=np.binary_repr(ord(name.split('.')[0]))[2:]#убираем 10 из начала, тк везде одинаково, то понижаем размерность
    #ys=int(ys)
    #ys1=np.binary_repr(ord(name.split('.')[0])).split(
    #ys1=np.binary_repr(ord(name.split('.')[0]))
    ys = ' '.join(ys)
    ys=np.array(ys.split(),dtype=int)

    #print(type(ys1))
    #ys=letter_to_bin(name.split('.')[0])
    #ys=(np.array(int(ys1)))
    #дополнить нулями слева
    #print(ys)
    #print(type(ys))
    #print(chr((ord(name.split('.')[0]))))
    if D is None:
        D = xs
        Y=ys
    else:
        D = np.vstack((D, xs))
        Y = np.vstack((Y, ys))
Y = np.swapaxes(Y, 0, 1)

#print(Y)

#print(D)
#print('---')
#print(len(Y))

for i in range(np.shape(w)[0]):
    while train(i):
        print(w[i])

path_test = '2_2_test/'




    #print(img)
for i in np.arange(0,90,10):

    for name in os.listdir(path_test):
        pos = 0
        neg = 0
        img = plt.imread('{}{}'.format(path_test, name))
        rotated = ndimage.rotate(img, i, reshape=0)
        ac = np.binary_repr(ord(name.split('.')[0]))[2:]
        #print("Бинарный код тестовой буквы 10",ac)
        xs = np.dot(img[..., :3], [1, 1, 1]) .flatten()  #Привели к ч/б умножив срез на [1, 1, 1]+ превращаем матрицу в вектор
        result = net(xs)
        #print("Бинарный код распознанной буквы 10",result)

        if result == ac:
            pos += 1
        else:
            neg += 1
        print("Угол поворота", i)
        print("Корректно распознал", pos / (pos + neg) * 100)
        print("Ошибочно распознал", neg / (pos + neg) * 100)


