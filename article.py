import matplotlib.pyplot as plt
import statistics
import numpy as np
import math
import hmm
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
def goglobal(self,obs):
	t = 10 - int(obs*10-12)
	return t
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

n = 88;
x = range(n)
y = np.loadtxt('raw_data/proc_bezrobitta.txt',delimiter='\t')
h= max(y)
l= min(y)
plt.plot(x, y,color='b')
second_model=hmm.SHMM(5,5,y,standart_instate,standart_instate)
second_model.show()
second_model.Baum_Welch()
second_model.show()
second_model.print("article.txt")
#second_model.print("files/test_model2.txt")
print("Prediction")
for i in [0,1,2,3,4]:
	print("State ", i,"P(Crisis) = ", second_model.PredictCrisis(4,i))

plt.show()	