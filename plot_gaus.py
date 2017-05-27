import numpy as np
import math
import matplotlib.pyplot as plt

def Norm(x,mu,sig):
	return math.exp(-(x-mu)**2/(2*sig**2))/((2*math.pi)**0.5*sig)

x = [0.1*i for i in range(100)]

ax = plt.subplot(2, 2, 1)
y = list(map(lambda x: 0.5*Norm(x,4,1)+0.5*Norm(x,6,1),x))
plt.plot(x, y, label="0.5N(4,1)+0.5N(6,1)",color='b')
plt.legend(loc='upper left',fancybox=True, shadow=True)

ax = plt.subplot(2, 2, 2)
y = list(map(lambda x: 0.3*Norm(x,3,2)+0.7*Norm(x,7,0.5),x))
plt.plot(x, y, label="0.3N(3,2)+0.7(7,0.5)",color='b')
plt.legend(loc='upper left',fancybox=True, shadow=True)

ax = plt.subplot(2, 2, 3)
y = list(map(lambda x: 0.5*Norm(x,3,1)+0.5*Norm(x,7,1),x))
plt.plot(x, y, label="0.5N(3,1)+0.5N(7,1)",color='b')
plt.legend(loc='upper left',fancybox=True, shadow=True)

ax = plt.subplot(2, 2, 4)
y = list(map(lambda x: 0.7*Norm(x,3,2)+0.3*Norm(x,7,0.5),x))
plt.plot(x, y, label="0.7N(3,2)+0.3(7,0.5)",color='b')
plt.legend(loc='upper left',fancybox=True, shadow=True)
plt.show()