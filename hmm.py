# -*- encoding: utf-8 -*-
import numpy as np
import itertools
import statistics
import math
from abc import ABCMeta, abstractmethod
def binom_instate(self,obs):
		if (obs==1):
			return 1
		return 0
def normalize(a):
	return list(map(lambda x: x/sum(a),a)) 
def main():
	N=100
	obs = np.random.binomial(1,0.35,N)
	model = HMM(2,2,obs,binom_instate,binom_instate)
	model.Baum_Welch()
	model.show()
	smodel = MM(2,obs,binom_instate)
	smodel.find_probs()
	smodel.show()
	tmodel = SHMM(2,2,obs,binom_instate,binom_instate)
	tmodel.Baum_Welch()
	tmodel.show()
	tmodel.Viterbi(5,1)

class MM():
	__metaclass__ = ABCMeta
	def __init__(self,n,ob,func):
		self.N = n
		self.T = len(ob)
		self.obs = ob
		self.A = [[1./self.N for i in range(self.N)] for j in range(self.N)]
		self.Pi = [1./self.N for i in range(self.N)]
		self.instate = func
		self.mean = statistics.mean(self.obs)
		self.stdev = statistics.stdev(self.obs)
	
	def PredictCrisis(self,n,current):
		s=0
		for i in range(n-1):
			s+=self.PredictCrisisN(i,current)
		return s
	def PredictCrisisN(self, n,current):
		notCrisis = [1,2,3,4]
		combinations = itertools.product(notCrisis,repeat=n)
		s=0
		for com in combinations:
			item = list(com)
			item.insert(0,current)
			item.append(0)
			s+=self.obsProb(item)
		return s

	def find_probs(self):
		for i in range(self.N):
			for j in range(self.N):
				s1=s2=0
				for t in range(self.T-1):
					if (((self.instate(self,self.obs[t]))==i)&((self.instate(self,self.obs[t+1]))==j)):
						s1+=1
					if (self.instate(self,self.obs[t]))==i:
						s2+=1	
				self.A[i][j] = s1/s2
	
	def obsProb(self,obs):
		d=1
		for i in range(1,len(obs)):
			d*=self.A[obs[i-1]][obs[i]]
		return d
	
	def show(self):
		print("Simple markov model:")
		print('A:\n',self.A)
		print('Pi:\n',self.Pi)
        		
class HMM(MM):
	def __init__(self, n , l, ob,funcA,funcB):
		super(HMM,self).__init__(n,ob,funcA)
		self.L = l
		self.B = [[ self.initB(i,j) for i in range(self.N)]for j in range(self.L)]
		self.obsTo=funcB		       
	def initB(self,i,j):
		if i==j:
			return .8
		else:
			return .2/(self.N-1)


	def show(self):
		print("hidden markov model:") 
		print('PI:\n',self.Pi)
		print('A:\n',self.A)
		print('B:\n',self.B)

	def Forward(self):
		alfa = [[self.Pi[i]*self.B[i][self.obsTo(self,self.obs[0])] if t==0 else 0 for i in range(self.N)] for t in range(self.T)]
		for t in range(1,self.T):
			for i in range(self.N):
				for j in range(self.N):
					alfa[t][i]+=alfa[t-1][j]*self.A[j][i]*self.B[i][self.obsTo(self,self.obs[t])]
		self.alfa = alfa

	def BackWard(self):
		beta = [[1 if j==self.T-1 else 0 for i in range(self.N)] for j in range(self.T)]
		for t in range(self.T-1,0,-1):
			for i in range(self.N):
				for j in range(self.N):
					beta[t-1][i]+=beta[t][j]*self.A[i][j]*self.B[j][self.obsTo(self,self.obs[t])]
		self.beta = beta;

	def perevirka(self,ksi,gama):
		print("perevirka:")
		for t in range(self.T-1):
			print("t",t)
			print("gama = ", gama[t])
			print("ksi ",ksi[t])
			
	def Baum_Welch(self):
		for iter in range(200):
			self.Forward()
			self.BackWard()
			ksi = [[[self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s] for q in range(self.N)]for s in range(self.N)] for t in range(self.T-1)]
			for t in range(self.T-1):
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s]

			gama  = [[self.beta[t][i]*self.alfa[t][i] for i in range(self.N)] for t in range(self.T)]
			pro = 0
			for i in range(self.N):
				pro+=self.alfa[self.T-1][i]
			for t in range(self.T):
				s1=0
				for j in range(self.N):
					s1+=self.beta[t][j]*self.alfa[t][j]
				for i in range(self.N):
					gama[t][i] = gama[t][i]/s1
			for t in range(self.T-1):
				s2=0
				for q in range(self.N):
					for s in range(self.N):
						s2+= self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s]
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = ksi[t][q][s]/s2
				
			for i in range(self.N):
				self.Pi[i] = gama[0][i];
			for i in range(self.N):			
				for j in range(self.N):
					s2 = 0
					for t in range(self.T-1):
						s2+=ksi[t][i][j]
					self.A[i][j] = s2

			for i in range(self.N):
				s1=0
				for t in range(self.T-1):
						s1+=gama[t][i]
				for j in range(self.N):
					self.A[i][j] =self.A[i][j]/s1

			for j in range(self.N):
				for k in range(self.L):
					s1=0
					s2=0
					for t in range(self.T):
						s1+=gama[t][j]
						if (self.obs[t] == k): s2+=gama[t][j]
					self.B[j][k] = s2/s1
class SHMM(MM):
	def __init__(self, n , l, ob,funcA,funcB):
		super(SHMM,self).__init__(n,ob,funcA)
		self.L = l
		self.B = [[ self.initB(i,j) for i in range(self.N)]for j in range(self.L)]
		self.obsTo=funcB		       
	def initB(self,i,j):
		if i==j:
			return .7
		else:
			return .3/(self.N-1)

	def show(self):
		print("hidden markov model:") 
		print('PI:\n',self.Pi)
		print('A:\n',self.A)
		print('B:\n',self.B)

	def Forward(self):
		alfa = [[self.Pi[i]*self.B[i][self.obsTo(self,self.obs[0])] if t==0 else 0 for i in range(self.N)] for t in range(self.T)]
		for t in range(1,self.T):
			for i in range(self.N):
				for j in range(self.N):
					alfa[t][i]+=alfa[t-1][j]*self.A[j][i]*self.B[i][self.obsTo(self,self.obs[t])]
		self.Falfa = alfa

	def BackWard(self):
		beta = [[1 if j==self.T-1 else 0 for i in range(self.N)] for j in range(self.T)]
		for t in range(self.T-1,0,-1):
			for i in range(self.N):
				for j in range(self.N):
					beta[t-1][i]+=beta[t][j]*self.A[i][j]*self.B[j][self.obsTo(self,self.obs[t])]
		self.Bbeta = beta;
	def FindPro(self):
		self.Pro = []
		for t in range(self.T):
			pro=0
			for i in range(self.N):
				pro+=self.Falfa[t][i]
			self.Pro.append(pro)
	def SForward(self):
		self.Forward()
		self.FindPro()
		self.alfa =  [[self.Falfa[t][i]/self.Pro[t] for i in range(self.N)] for t in range(self.T)]
	
	def SBackWard(self):
		self.BackWard()
		self.beta =  [[self.Bbeta[t][i]/self.Pro[self.T-1-t] for i in range(self.N)] for t in range(self.T)]

	def FindBest(self,i,t,d):
		a = [ d[j][t-1]*self.A[j][i] for j in range(self.N)]
		return (np.argmax(a),np.max(a))

	def Viterbi(self, K, obs):
		d = [[self.Pi[i]*self.B[i][self.obsTo(self,obs)]] for i in range(self.N)]
		psi = [self.obsTo(self,obs)]
		for t in range(K):
			for i in range(self.N):
				arg,m = self.FindBest(i,t,d)
				psi.append(arg)
				d[i].append(m*self.B[i][psi[-1::][0]])
		return psi


	def Baum_Welch(self):
		for iter in range(50):
			self.SForward()
			self.SBackWard()

			ksi = [[[self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s] for q in range(self.N)]for s in range(self.N)] for t in range(self.T-1)]
			for t in range(self.T-1):
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s]

			gama  = [[self.beta[t][i]*self.alfa[t][i] for i in range(self.N)] for t in range(self.T)]
			pro = 0
			for i in range(self.N):
				pro+=self.alfa[self.T-1][i]
			for t in range(self.T):
				s1=0
				for j in range(self.N):
					s1+=self.beta[t][j]*self.alfa[t][j]
				for i in range(self.N):
					gama[t][i] = gama[t][i]/s1
			for t in range(self.T-1):
				s2=0
				for q in range(self.N):
					for s in range(self.N):
						s2+= self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s]
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = ksi[t][q][s]/s2
				
			for i in range(self.N):
				self.Pi[i] = gama[0][i];
			for i in range(self.N):			
				for j in range(self.N):
					s2 = 0
					for t in range(self.T-1):
						s2+=ksi[t][i][j]
					self.A[i][j] = s2

			for i in range(self.N):
				s1=0
				for t in range(self.T-1):
						s1+=gama[t][i]
				for j in range(self.N):
					self.A[i][j] =self.A[i][j]/s1

			for j in range(self.N):
				for k in range(self.L):
					s1=0
					s2=0
					for t in range(self.T):
						s1+=gama[t][j]
						if (self.obsTo(self,self.obs[t]) == k): s2+=gama[t][j]
					self.B[j][k] = s2/s1
class CGMHMM(MM):
	def __init__(self, n , ob,funcA,mu,sig,w):
		super(CGMHMM,self).__init__(n,ob,funcA)
		self.M = len(mu)
		self.mu = mu
		self.sig = sig
		self.w = w
	def B(self,j,x):
		s=0
		for k in range(self.M):
			s+=self.w[k][j]*self.GuausB(j,k,x)
		return s
	def GuausB(self,j,k,x):
		res =  math.exp(-(x-self.mu[k][j])**2/(2*self.sig[k][j]**2))/((2*math.pi)**0.5*self.sig[k][j])
		if math.isnan(res): print(res , x,self.mu[k][j],self.sig[k][j],k,j)
		return res
	def show(self):
		print("continious gausian mixture:")
		print('PI:\n',self.Pi)
		print('A:\n',self.A)
		print('mu:\n',self.mu)
		print('sigma:\n',self.sig)
		print('w:\n',self.w)

	def Forward(self):
		alfa = [[self.Pi[i]*self.B(i,self.obs[0]) if t==0 else 0 for i in range(self.N)] for t in range(self.T)]
		for t in range(1,self.T):
			for i in range(self.N):
				for j in range(self.N):
					alfa[t][i]+=alfa[t-1][j]*self.A[j][i]*self.B(i,self.obs[t])
		self.Falfa = alfa

	def BackWard(self):
		beta = [[1 if j==self.T-1 else 0 for i in range(self.N)] for j in range(self.T)]
		for t in range(self.T-1,0,-1):
			for i in range(self.N):
				for j in range(self.N):
					beta[t-1][i]+=beta[t][j]*self.A[i][j]*self.B(j,self.obs[t])
		self.Bbeta = beta;
	def FindPro(self):
		self.Pro = []
		for t in range(self.T):
			pro=0
			for i in range(self.N):
				pro+=self.Falfa[t][i]
			self.Pro.append(pro)
	def SForward(self):
		self.Forward()
		self.FindPro()
		#print(self.Pro)
		#self.alfa = self.Falfa
		self.alfa =  [[self.Falfa[t][i]/self.Pro[t] for i in range(self.N)] for t in range(self.T)]
	
	def SBackWard(self):
		self.BackWard()
		#self.beta = self.Bbeta
		self.beta =  [[self.Bbeta[t][i]/self.Pro[self.T-1-t] for i in range(self.N)] for t in range(self.T)]

	def FindBest(self,i,t,d):
		a = [ d[j][t-1]*self.A[j][i] for j in range(self.N)]
		return (np.argmax(a),np.max(a))

	def Viterbi(self, K, obs):
		d = [[self.Pi[i]*self.B(i,obs)] for i in range(self.N)]
		psi = [self.obsTo(self,obs)]
		for t in range(K):
			for i in range(self.N):
				arg,m = self.FindBest(i,t,d)
				psi.append(arg)
				d[i].append(m*self.B(i,psi[-1::][0]))
		return psi

	def Crisis(self,n,obs):
		s=0
		prob = []
		for q in range(self.N):
			prob.append(self.B(q,obs))
		prob = normalize(prob)
		for q in range(self.N):
			for i in range(n-1):
				s+=prob[q]*self.PredictCrisisN(i,q)
		return s

	def Baum_Welch(self):
		for iter in range(20):
			self.SForward()
			self.SBackWard()

			ksi = [[[self.alfa[t][q]*self.A[q][s]*self.B(s,self.obs[t+1])*self.beta[t+1][s] for q in range(self.N)]for s in range(self.N)] for t in range(self.T-1)]
			for t in range(self.T-1):
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = self.alfa[t][q]*self.A[q][s]*self.B(s,self.obs[t+1])*self.beta[t+1][s]

			gama  = [[self.beta[t][i]*self.alfa[t][i] for i in range(self.N)] for t in range(self.T)]
			for t in range(self.T):
				s1=0
				for j in range(self.N):
					s1+=self.beta[t][j]*self.alfa[t][j]
				for i in range(self.N):
					gama[t][i] = gama[t][i]/s1
			for t in range(self.T-1):
				s2=0
				for q in range(self.N):
					for s in range(self.N):
						s2+= self.alfa[t][q]*self.A[q][s]*self.B(s,self.obs[t+1])*self.beta[t+1][s]
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = ksi[t][q][s]/s2
				
			for i in range(self.N):
				self.Pi[i] = gama[0][i];
			
			for i in range(self.N):			
				for j in range(self.N):
					s2 = 0
					for t in range(self.T-1):
						s2+=ksi[t][i][j]
					self.A[i][j] = s2
			#print(iter, self.A)
			for i in range(self.N):
				s1=0
				for t in range(self.T-1):
						s1+=gama[t][i]
				for j in range(self.N):
					self.A[i][j] =self.A[i][j]/s1
			#print(iter,ksi[-10::],"\nalfa",self.alfa[-10::],"\nbeta",self.Bbeta[-10::],self.beta[-10::])
			#print(iter, self.A)
			#ngama = [[[gama[t][i]*self.w[k][i]*self.GuausB(i,k,self.obs[t])/self.B(i,self.obs[t]) for i in range(self.N)]for k in range(self.M)] for t in range(self.T)]
			ngama = [[[gama[t][i] for i in range(self.N)]for k in range(self.M)] for t in range(self.T)]
			
			for i in range(self.N):
				for k in range(self.M):
					s1=s2=s3=s4=0
					for t in range(self.T):
						s1+=gama[t][i]
						s2+=ngama[t][k][i]
						s3+=ngama[t][k][i]*self.obs[t]
						s4+=ngama[t][k][i]*(self.obs[t]-self.mu[k][i])**2
					self.mu[k][i] = s3/s2
					self.w[k][i] = s2/s1
					self.sig[k][i] = s4/s2
					if self.sig[k][i]<0.3:self.sig[k][i] = 0.3
			#print(iter, self.Pro[self.T-1])
			#if iter==1: self.show()
if __name__ == '__main__':
	main()