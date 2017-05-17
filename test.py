import matplotlib.pyplot as plt
import statistics
import numpy as np
import math
import hmm

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
def standart_means(y):
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	ymax = max(y)
	ymin = min(y)
	#return [(ymax+mean+stdev)/2,mean,(mean-stdev+ymin)/2]
	return [(ymax+mean+stdev)/2,(mean+stdev+mean+0.5*stdev)/2,mean,(mean-stdev+mean-0.5*stdev)/2,(mean-stdev+ymin)/2]

def main():
	N=365
	x = range(N)
	eps = np.random.binomial(2,0.3,N)
	y = eps+np.sin(list(map(lambda x: x/2,x)))+np.sin(list(map(lambda x: x/3,x)))+np.sin(list(map(lambda x: x/5,x)))+np.sin(list(map(lambda x: x/7,x)))
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	crisis = list(map(lambda x: isCrisis(x,mean,stdev),y))
	margins = getCrisisMargins(crisis)
	mainplot = plt.plot(x, y)
	for i in margins:
		tmpx = []
		tmpy = []
		for item in range(i[0],i[1]):
			tmpy.append(y[item])
			tmpx.append(item)
		plt.fill_between(tmpx,min(y),max(y),alpha=0.3, color='red')
	
	first_model=hmm.MM(5,y,standart_instate)
	first_model.find_probs()
	#first_model.show()
	for i in [0,1,2,3,4]:
		print(i,first_model.PredictCrisis(4,i))

	second_model=hmm.SHMM(5,5,y,standart_instate,standart_instate)
	#second_model.Baum_Welch()
	#second_model.show()
	#for i in [0,1,2,3,4]:
		#print(i,second_model.PredictCrisis(4,i))
	st_mu = standart_means(y)
	
	#plt.setp(mainplot, color='r', linewidth=2.0)
	#addplot = plt.plot(x, np.full(N,st_mu[0]),x, np.full(N,st_mu[1]),x, np.full(N,st_mu[2]),x, np.full(N,st_mu[3]),x, np.full(N,st_mu[4]))
	#plt.setp(addplot, color='g', linewidth=2.0)
	mu = [st_mu]
	sig = [list(np.full(5,0.6))]
	w = [list(np.full(5,1))]
	
	model = hmm.CGMHMM(5,y[:100:],standart_instate,mu,sig,w)
	model.Baum_Welch()
	for i in [0,1,2,3,4]:
		print(i,model.PredictCrisis(4,i))
	s=-2
	for i in range(70):
		print("s = ",s)
		print("crisis = ",model.Crisis(5,s))
		s+=0.1
	first_model.show()
	#second_model.show()
	model.show()
	#plt.show()


if __name__ == '__main__':
	main()