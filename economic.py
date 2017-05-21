import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import math
import hmm

def showCrisisDistr(model,y,eps):
	s=min(y)
	t=[]
	x=[]
	while s<max(y):
		t.append(s)
		x.append(model.Crisis(5,s))
		s+=eps
	plt.plot(t, x, label = "state"+str(i))
	plt.legend()
	plt.show()

def isCrisis(x,m,s):
	return(x>m+s)

def getCrisisMargins(y):
	a = []
	for i in range (len(y)-1):
		if y[i]!=y[i+1]:
			a.append(i+1)
	res = [[a[i],a[i+1]] for i in range(0,len(a)-1,2)]
	return res

def standart_instate(self,obs):
	mean=self.mean
	stdev= self.stdev
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

def standart_means(y):
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	ymax = max(y)
	ymin = min(y)
	return [(ymax+mean+stdev)/2,(mean+stdev+mean+0.5*stdev)/2,mean,(mean-stdev+mean-0.5*stdev)/2,(mean-stdev+ymin)/2]

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
def getcolor(i):
	if i==0:
		return '#cc0099'
	elif i==1:
		return '#3399ff'
	elif i==2:
		return '#99ff00'
	elif i==3:
		return 'grey'
	else:
		return 'silver'
def compute(arr,c):
	res = [0 for i in range(c)]
	for i in range(c,len(arr)):
		res.append(arr[i]-arr[i-c])
	return res
def main():
	y = np.loadtxt('data2.txt',delimiter='\t')
	y1 = np.loadtxt('reservs893.data',delimiter='\t')
	y2 = np.loadtxt('bezr893.data',delimiter='\t')
	y3 = np.loadtxt('borg893.data',delimiter='\t')
	y4 = np.loadtxt('procent893.data',delimiter='\t')
	data = []
	N  = len(y)

	first_model=hmm.MM(5,y,standart_instate)
	first_model.find_probs()
	#
	second_model=hmm.SHMM(5,5,y[-100::],standart_instate,standart_instate)
	second_model.Baum_Welch()	
	#
	st_mu = standart_means(y[-100::])
	
	mu = [list(map(lambda x: x-0.1,st_mu)),list(map(lambda x: x+0.1,st_mu))]
	sig = [list(np.full(5,0.6)),list(np.full(5,0.6))]
	w = [list(np.full(5,0.5)),list(np.full(5,0.5))]
	#
	model = hmm.CGMHMM(5,y[-100::],standart_instate,mu,sig,w)
	model.Baum_Welch()

	
	first_model.show()
	second_model.show()
	model.show()
	
if __name__ == '__main__':
	main()