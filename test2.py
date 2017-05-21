import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import math
import hmm

def yinstate(obs,y):
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	if obs>mean+stdev:
		return 0
	elif obs>mean+0.5*stdev:
		return 1
	elif obs>mean-0.5*stdev:
		return 2
	elif obs>mean-stdev:
		return 3
	else:
		return 4

def getMargins(state, y):
	a = []
	res = []
	y = list(map(lambda x: yinstate(x,y)==state,y))
	for i in range (1,len(y)):
		if y[i]:
			if i==1: a.append(i)
			elif y[i-1]==False: a.append(i)
			if i==len(y)-1 or y[i+1]==False:
				a.append(i)
				res.append(a)
				a=[]
	return res

def main():
	y = np.loadtxt('data2.txt',delimiter='\t')
	N = len(y)
	x= range(N)
	
	plt.plot(x, y,label="y", color='b')
	plt.xlabel(u'time')
	plt.ylabel(u'index')
	plt.title(u'Тестові дані')
	first = True
	margins = getMargins(0,y)
	for i in margins:
		if first: 
			first = False
			plt.fill_between(i,min(y),max(y),alpha=0.3,label="Кризовий стан", color='r')
		else: plt.fill_between(i,min(y),max(y),alpha=0.3, color='red')
	plt.legend(loc='upper right',fancybox=True, shadow=True, ncol=3)
	plt.show()



if __name__ == '__main__':
	main()