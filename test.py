import matplotlib.pyplot as plt
import statistics
import numpy as np
import math

def isCrisis(x,m,s):
	return(x>m+s)

def getCrisisMargins(y):
	a = []
	for i in range (len(y)-1):
		if y[i]!=y[i+1]:
			a.append(i+1)
	res = [[a[i],a[i+1]] for i in range(0,len(a)-1,2)]
	return res

x = range(365)
y = np.sin(list(map(lambda x: x/2,x)))+np.sin(list(map(lambda x: x/3,x)))+np.sin(list(map(lambda x: x/5,x)))+np.sin(list(map(lambda x: x/7,x)))

mean = statistics.mean(y)
stdev = statistics.stdev(y)

crisis = list(map(lambda x: isCrisis(x,mean,stdev),y))
margins = getCrisisMargins(crisis)
plt.plot(x, y, x, np.full(365,mean+stdev))
for i in margins:
	tmpx = []
	tmpy = []
	for item in range(i[0],i[1]):
		tmpy.append(y[item])
		tmpx.append(item)
	plt.fill_between(tmpx,min(y),tmpy,alpha=0.3, color='red')
plt.show()