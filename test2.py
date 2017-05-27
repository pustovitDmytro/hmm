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
def normdata(i,x):
	if min(x)==max(x): return 0
	return (x[i]-min(x))/(max(x)-min(x))
def GuausB(x,mu,sig):
	mu = np.asarray(mu).reshape(-1)
	#print("x=",x,"\nsig=",self.sig[k][j],"\nmu=",mu)
	x= np.matrix(x)
	x=x.T
	mu= np.matrix(mu)
	mu=mu.T
	sig = np.matrix(sig)
	res =  math.exp(-0.5*(x-mu).T*sig.I*(x-mu))/((2*math.pi)**0.5*np.linalg.det(sig)**0.5)
	print((x-mu).T*sig.I*(x-mu))
	print(res)
def main():
	x= np.matrix([0.83647798742138357, 0.20276243901725813, 1.0, 0.76406025824964074, 0.0])
	mu= np.matrix([ 0.76797675,  0.55465737,  0.74648533,  0.77541036,  1.00100756])
	sig = np.matrix([[0.6 if i==j else 0 for i in range(5)]for j in range(5)])
	GuausB(x,mu,sig)
	
if __name__ == '__main__':
	main()