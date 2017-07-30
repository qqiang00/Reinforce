from matplotlib import pyplot as plt
import math
import numpy as np
from pylab import *
from random import random

D_INCREMENT = 1
# 输入可以是 (-inf, +inf)
# 输出最多为 [0, 1] 不为负值
# 阈值不限定,对于神经元来说，一般同一类神经元的阈值是固定不变的。
# 权重  weight [-1, 1]
# 持续反应系数 [0,1]
class Neuron(object):
    def __init__(self):
        self.set(a=0.2, b=-0.2, w=0.9)
        self.output = 0.0
    
    def _compute_output(self, x):
        self.output = self.a * self.output + D_INCREMENT * (self.w * x + self.b > 0)
        #if self.output >= 1.0:
        #    self.output = 1.0
        return self.output

    def __call__(self, x):
        return self._compute_output(x)

    def copy(self):
        new_n = Neuron()
        new_n.a = self.a
        new_n.b = self.b
        new_n.w = self.w
        return new_n

    def constrains(self):
        if self.a >= 1: self.a = 0.99
        elif self.a <= 0: self.a = 0
        if self.w <= 0: self.w = 0.0
        elif self.w >= 1: self.w = 1.0

    def set(self, a:float, b:float, w:float):
        self.a, self.b, self.w = a, b, w
        self.constrains()


def main():
    sample_num = 256
    t = np.linspace(0, 2*np.pi, sample_num, endpoint=True)
    #t = np.linspace(0, 1, 20, endpoint=True)
    x = 2*np.sin(t)
    #x = np.array([2 for _ in t])
    #x = np.array([2*(random() > 0.98) for i in range(sample_num)])
    n1 = Neuron()
    n2 = n1.copy()
    n1.set(a=0.1, b=-1.1, w=0.6)
    n2.set(a=0.8, b = -1.1, w = 0.6 )
    y1 = np.array([n1(t) for t in x])
    y2 = np.array([n2(t) for t in x])
    #print(x)
    print(y1)
    print(y2)

    plot(t,x/40)
    plot(t,y1)
    plot(t,y2)
    show()
    pass


if __name__ == "__main__":
    main()