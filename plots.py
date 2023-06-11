# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:09:00 2020

@author: singh
"""
%matplotlib qt5

from matplotlib import pyplot as plt
import numpy as np

def power(x, power_fact):
    '''Supports both positive and negative integer powers
    '''
    n = x
    i_range = power_fact-1 
    if i_range == -1:
        return 1
    if i_range < -1:
        for i in range(-i_range):
            x /= n
        return x
    if i_range >= 0:
        for i in range(i_range):
            x *= n 
        return x
    
    
    
    
    

def factorial(x):
    fact = 1
    if x in [0, 1]:
        return 1
    else:
        for i in range(x, 1, -1):
            fact *= i
        return fact

y = factorial(5)
y = power(2, -3)
def parabola(x):
    
    y = x**(2)
    
    return y

def sin(x):
    '''Implementation from Taylor series
    sin(x) => summation[(-1)^k * x^(2k + 1)/(2k + 1)!] 
             k = [0, inf) 
    x in degrees
    '''
    # Degree conversion to radians
    pi = 3.1415926535
    x /= 180
    x *= pi
    
    _sum=0
    for k in range(60):
        n = power(-1, k) * power(x, 2*k + 1) / factorial(2*k + 1)
        _sum += n
    
    return _sum
    
y = sin(45)



X = np.arange(-10, 11)

X_deg = np.arange(-360, 360.1, 0.1)

Y = np.array([])

Y_deg = np.array([])


for x in X_deg:
    y = sine(x)
    Y_deg = np.append(Y_deg, [y])




Y_max = np.max(Y_deg)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim((-Y_max, Y_max))
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.title("Sine function")
plt.xticks([-360, -270, -180, -90, 0, 90, 180, 270, 360])
plt.plot(X_deg, Y_deg)
plt.show()


