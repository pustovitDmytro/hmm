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
	
def warningSystem(model, num_days=5, error=0.0):
	arr=[]
	prob = []
	y = model.obs	
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
	N=365
	x = range(N)
	#GENERATE NEW DATA
	#eps = np.random.binomial(2,0.3,N)
	#y = eps+np.sin(list(map(lambda x: x/2,x)))+np.sin(list(map(lambda x: x/3,x)))+np.sin(list(map(lambda x: x/5,x)))+np.sin(list(map(lambda x: x/7,x)))
	#np.savetxt('test_y.txt',y,delimiter='\t',fmt='%1.8e')
	
	y = np.loadtxt('test_y.txt',delimiter='\t')
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)

	plt.plot(x, y,color='b')
	plt.xlabel(u'time')
	plt.ylabel(u'index')
	plt.title(u'Тестові дані')

	first = True
	margins = getMargins(0,y)
	for i in margins:
		if first: 
			first = False
			plt.fill_between(i,min(y),max(y),alpha=0.3,label="Crisis", color="red")
		else: plt.fill_between(i,min(y),max(y),alpha=0.3, color='red')
	plt.legend(loc='upper right',fancybox=True, shadow=True)
	plt.show()
	plt.clf()

	#for state in [0,1,2,3,4]:
	#	first = True
	#	margins = getMargins(state,y)
	#	for i in margins:
	#		if first: 
	#			first = False
	#			plt.fill_between(i,min(y),max(y),alpha=0.3,label="Δ"+str(state+1), color=getcolor(state))
	#		else: plt.fill_between(i,min(y),max(y),alpha=0.3, color=getcolor(state))
	#plt.legend(loc='upper center',fancybox=True, shadow=True, ncol=6)
	#plt.show()
	#plt.clf()

	first_model=hmm.MM(5,y,standart_instate)
	first_model.find_probs()
	first_model.show()
	first_model.print("files/test_model1.txt")
	print("Prediction")
	for i in [0,1,2,3,4]:
		print("State ", i,"P(Crisis) = ", first_model.PredictCrisis(4,i))
	plt.plot(x, y,color='b')
	plt.title(u'Model_1')
	warningSystem(first_model,num_days = 5, error=0.10)


	second_model=hmm.SHMM(5,5,y,standart_instate,standart_instate)
	second_model.Baum_Welch()
	second_model.show()
	second_model.print("files/test_model2.txt")
	print("Prediction")
	for i in [0,1,2,3,4]:
		print("State ", i,"P(Crisis) = ", first_model.PredictCrisis(4,i))
	plt.plot(x, y,color='b')
	plt.title(u'Model_2')
	warningSystem(second_model,num_days = 5, error=0.05)
	
	st_mu = standart_means(y)
	mu = [list(map(lambda x: x-0.1,st_mu)),list(map(lambda x: x+0.1,st_mu))]
	sig = [list(np.full(5,0.6)),list(np.full(5,0.6))]
	w = [list(np.full(5,0.5)),list(np.full(5,0.5))]
	third_model = hmm.CGMHMM(5,y,standart_instate,mu,sig,w)
	third_model.Baum_Welch()
	third_model.show()
	third_model.print("files/test_model3.txt")
	print("Prediction(classic)")
	for i in [0,1,2,3,4]:
		print("State ", i,"P(Crisis) = ", third_model.PredictCrisis(4,i))
	plt.plot(x, y,color='b')
	plt.title(u'Model_3')
	warningSystem(third_model,num_days = 5, error=0.10)


	names = ['A','B','C','D','E',"F","G"]
	x_range = [93,95,97,98,99,100,104]
	plt.plot(x, y,color='b')
	first = True
	margins = getMargins(0,y)
	for i in margins:
		if first: 
			first = False
			plt.fill_between(i,min(y),max(y),alpha=0.3,label="Crisis", color="red")
		else: plt.fill_between(i,min(y),max(y),alpha=0.3, color='red')
	plt.legend(loc='upper right',fancybox=True, shadow=True)
	for i in range(7):
		plt.annotate(names[i], xy=(x_range[i], y[x_range[i]]), xytext=(x_range[i]-5,y[x_range[i]]), arrowprops=dict(facecolor='black',width=0.5, headwidth=0.5))
		print(names[i])
		state = standart_instate(first_model,y[x_range[i]])
		print("first", first_model.PredictCrisis(4,state))
		print("second", second_model.PredictCrisis(4,state))
		print("third", third_model.PredictCrisis(4,state))
		print("third2", third_model.Crisis(5,y[x_range[i]]))
	plt.show()

if __name__ == '__main__':
	main()