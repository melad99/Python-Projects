'HW#2-Q1'
'Student: Melad ALsaleh'
'ID: 120170346'

import numpy as np
import matplotlib.pyplot as plt

M = 8 
N = 8
L = 4

n =[10, 20, 10, 24]
r = [0,1,2,3]
T= [0,0,0,0]

for i in range(len(n)):
    n[i] = n[i]/(M*N)

for j in range(len(n)):
    T[j] = ((L-1)/M*N)*n[j]

# plotting the histogram
plt.stem(r, n)
plt.xlabel('intensity value')
plt.ylabel('P(T(rk))')
plt.title('equalized Histogram')
plt.show()

