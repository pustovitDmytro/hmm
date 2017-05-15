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

def main():
	N=365
	x = range(N)
	y = np.sin(list(map(lambda x: x/2,x)))+np.sin(list(map(lambda x: x/3,x)))+np.sin(list(map(lambda x: x/5,x)))+np.sin(list(map(lambda x: x/7,x)))
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	crisis = list(map(lambda x: isCrisis(x,mean,stdev),y))
	margins = getCrisisMargins(crisis)
	plt.plot(x, y, x, np.full(N,mean+stdev),x,np.full(N,mean-stdev))
	for i in margins:
		tmpx = []
		tmpy = []
		for item in range(i[0],i[1]):
			tmpy.append(y[item])
			tmpx.append(item)
		plt.fill_between(tmpx,min(y),max(y),alpha=0.3, color='red')
	#plt.show()
	first_model=hmm.MM(5,y,standart_instate)
	first_model.find_probs()
	first_model.show()
	second_model=hmm.SHMM(5,5,y,standart_instate,standart_instate)
	second_model.Baum_Welch()
	second_model.show()



if __name__ == '__main__':
	main()