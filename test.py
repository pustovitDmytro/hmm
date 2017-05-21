import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
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
	y = list(map(lambda x: yinstate(x,y)==state,y))
	for i in range (len(y)-1):
		if y[i]!=y[i+1]:
			a.append(i+1)
	res = [[a[i],a[i+1]] for i in range(0,len(a)-1,2)]
	return res
def getcolor(i):
	if i==0:
		return '#cc0099'
	elif i==1:
		return '#3399ff'
	elif i==2:
		return '#99ff00'
	else:
		return 'white'
		
def main():
	N=365
	x = range(N)
	eps = np.random.binomial(2,0.3,N)
	#y = eps+np.sin(list(map(lambda x: x/2,x)))+np.sin(list(map(lambda x: x/3,x)))+np.sin(list(map(lambda x: x/5,x)))+np.sin(list(map(lambda x: x/7,x)))
	#np.savetxt('test1_y.txt',y,delimiter='\t',fmt='%1.8e')
	y = np.loadtxt('test1_y.txt',delimiter='\t')
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	crisis = list(map(lambda x: isCrisis(x,mean,stdev),y))
	margins = getCrisisMargins(crisis)
	mainplot = plt.plot(x, y,color='b')
	plt.xlabel(u'time')
	plt.ylabel(u'index')
	plt.title(u'Тестові дані')
	
	for i in margins:
		tmpx = []
		tmpy = []
		for item in range(i[0],i[1]):
			tmpy.append(y[item])
			tmpx.append(item)
		plt.fill_between(tmpx,min(y),max(y),alpha=0.3, color='red')
	
	first_model=hmm.MM(5,y,standart_instate)
	first_model.find_probs()
	#
	second_model=hmm.SHMM(5,5,y,standart_instate,standart_instate)
	second_model.Baum_Welch()	
	#
	st_mu = standart_means(y)
	#
	##plt.setp(mainplot, color='r', linewidth=2.0)
	##addplot = plt.plot(x, np.full(N,st_mu[0]),x, np.full(N,st_mu[1]),x, np.full(N,st_mu[2]),x, np.full(N,st_mu[3]),x, np.full(N,st_mu[4]))
	##plt.setp(addplot, color='g', linewidth=2.0)
	#
	mu = [list(map(lambda x: x-0.1,st_mu)),list(map(lambda x: x+0.1,st_mu))]
	sig = [list(np.full(5,0.6)),list(np.full(5,0.6))]
	w = [list(np.full(5,0.5)),list(np.full(5,0.5))]
	#
	model = hmm.CGMHMM(5,y,standart_instate,mu,sig,w)
	model.Baum_Welch()
	#Show predicition
	#for i in [0,1,2,3,4]:
	#	print(i,model.PredictCrisis(4,i))
	#for i in [0,1,2,3,4]:
		#print(i,first_model.PredictCrisis(4,i))
	#s=-2
	#for i in [0,1,2,3,4]:
	#	print(i,second_model.PredictCrisis(4,i))
	#for i in range(70):
	#	print("s = ",s)
	#	print("crisis = ",model.Crisis(5,s))
	#	s+=0.1
	#show models
	#first_model.show()
	#second_model.show()
	model.show()
	#plt.title(u'Неперервний прихований ланцюг Маркова з Гаусівськими компонентами')
	#for s in [0,1]:
	#	first = True
	#	prob = model.PredictCrisis(4,s)
	#	margins = getMargins(s,y)
	#	for i in margins:
	#		if first: 
	#			first = False
	#			plt.fill_between(i,min(y),max(y),alpha=0.3,label="P(Crisis)>"+str(round(prob,3)), color=getcolor(s))
	#		else: plt.fill_between(i,min(y),max(y),alpha=0.3, color=getcolor(s))
	#plt.legend(loc='upper center',fancybox=True, shadow=True, ncol=3)
	
	
	np.savetxt('test1_1.txt',model.A,delimiter='\t',fmt='%1.8e')
	
	names = ['A','B','C','D','E',"F","G"]
	x_range = [93,95,97,98,99,100,104]
	for i in range(7):
		plt.annotate(names[i], xy=(x_range[i], y[x_range[i]]), xytext=(x_range[i]-5,y[x_range[i]]), arrowprops=dict(facecolor='black',width=0.5, headwidth=0.5))
		print(names[i])
		state = standart_instate(first_model,y[x_range[i]])
		print(state)
		print("first", first_model.PredictCrisis(4,state))
		print("second", second_model.PredictCrisis(4,state))
		print("third", model.PredictCrisis(4,state))
		print("third2", model.Crisis(5,y[x_range[i]]))


	plt.legend(('y', 'кризові ситуації'),loc=1, frameon=True)
	plt.show()
	#plt.savefig('test Marcov', fmt='png')
	
	

if __name__ == '__main__':
	main()