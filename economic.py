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

def standart_means(data):
	data = np.matrix(data)
	data = data.T
	N = len(data)
	res = []
	for i in range(N):
		y = np.asarray(data[i]).reshape(-1)
		mean = statistics.mean(y)
		stdev = statistics.stdev(y)
		ymax = max(y)
		ymin = min(y)
		res.append([(ymax+mean+stdev)/2,(mean+stdev+mean+0.5*stdev)/2,mean,(mean-stdev+mean-0.5*stdev)/2,(mean-stdev+ymin)/2])
	res = np.matrix(res)
	res = res.T
	return res

def standart_mean(y):
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

def normdata(i,x):
	if min(x)==max(x): return 0
	return 2*(x[i]-min(x))/(max(x)-min(x))

def warningSystem(model, y, num_days=5, error=0.0):
	arr=[]
	prob = []
	for i in [0,1,2,3,4]:
		p = model.PredictCrisis(num_days-1,i)
		if p>error:
			arr.append(i)
			prob.append(p)
	for state in arr:
		first = True
		margins = getMargins(state,y)
		for i in margins:
			if first: 
				first = False
				plt.fill_between(i,min(y),max(y),alpha=0.3,label="P(Δ"+str(state+1)+")="+str(prob[state]), color=getcolor(state))
			else: plt.fill_between(i,min(y),max(y),alpha=0.3, color=getcolor(state))
	plt.legend(loc='upper center',fancybox=True, shadow=True, ncol=6)
	plt.show()
	plt.clf()

def main():
	y = np.loadtxt('real_data/data2.txt',delimiter='\t')
	y1 = np.loadtxt('real_data/reservs893.data',delimiter='\t')
	y2 = np.loadtxt('real_data/bezr893.data',delimiter='\t')
	y3 = np.loadtxt('real_data/borg893.data',delimiter='\t')
	y4 = np.loadtxt('real_data/procent893.data',delimiter='\t')
	data = []
	y_data = y[::-1]
	x_data = range(len(y))

	first_model=hmm.MM(5,y,standart_instate)
	first_model.find_probs()
	first_model.show()
	first_model.print("files/real_model1.txt")
	print("Prediction")
	for i in [0,1,2,3,4]:
		print("State ", i,"P(Crisis) = ", first_model.PredictCrisis(4,i))
	
	y = y[-100::]
	y1=y1[-100::]
	y2=y2[-100::]
	y3=y3[-100::]
	y4=y4[-100::]

	second_model=hmm.SHMM(5,5,y,standart_instate,standart_instate)
	second_model.Baum_Welch()
	second_model.show()
	second_model.print("files/real_model2.txt")
	print("Prediction")
	for i in [0,1,2,3,4]:
		print("State ", i,"P(Crisis) = ", second_model.PredictCrisis(4,i))
	plt.plot(x_data, y_data,color='b')
	plt.title(u'Model_2')
	warningSystem(second_model, y_data, num_days=5, error = 0.03)



	st_mu = standart_mean(y)
	mu = [list(map(lambda x: x-0.1,st_mu)),list(map(lambda x: x+0.1,st_mu))]
	sig = [list(np.full(5,0.6)),list(np.full(5,0.6))]
	w = [list(np.full(5,0.5)),list(np.full(5,0.5))]
	third_model = hmm.CGMHMM(5,y,standart_instate,mu,sig,w,dim=1)
	third_model.Baum_Welch()
	third_model.show()
	third_model.print("files/real_model3.txt")
	print("Prediction(classic)")
	for i in [0,1,2,3,4]:
		print("State ", i,"P(Crisis) = ", third_model.PredictCrisis(4,i))
	prob = third_model.Crisis(5,18.7)
	print("prob = ", prob)

	data = [[normdata(i,y),normdata(i,y1),normdata(i,y2),normdata(i,y3),normdata(i,y4)] for i in range(len(y))]
	st_mu = standart_means(data)

	mu = [list(map(lambda x: x-0.05,st_mu)),list(map(lambda x: x+0.05,st_mu))]
	SIGMA = np.matrix([[0.1 if i==j else 0 for i in range(5)]for j in range(5)])
	sig = [[SIGMA for i in range(5)],[SIGMA for i in range(5)]]
	w = [list(np.full(5,0.5)),list(np.full(5,0.5))]

	model = hmm.CGMHMM(5,data,standart_instate,mu,sig,w,dim=5)
	model.Baum_Welch()
	prob = model.Crisis(5,data[-1])
	np.savetxt('files/real_model4.txt',model.A,delimiter='\t',fmt='%1.4e')
	model.show()
	print("prob = ", prob)
	
	
if __name__ == '__main__':
	main()