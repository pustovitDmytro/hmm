# -*- encoding: utf-8 -*-
import numpy
import statistics
from abc import ABCMeta, abstractmethod
def binom_instate(self,obs):
		if (obs==1):
			return 1
		return 0
def main():
	N=100
	obs = numpy.random.binomial(1,0.35,N)
	model = HMM(2,2,obs,binom_instate,binom_instate)
	model.Baum_Welch()
	model.show()
	smodel = MM(2,obs,binom_instate)
	smodel.find_probs()
	smodel.show()
	tmodel = SHMM(2,2,obs,binom_instate,binom_instate)
	tmodel.Baum_Welch()
	tmodel.show()

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
		self.beta =  [[self.Bbeta[t][i]/self.Pro[t] for i in range(self.N)] for t in range(self.T)]

	def Baum_Welch(self):
		for iter in range(200):
			print(iter)
			self.show()
			self.SForward()
			self.SBackWard()
			ksi = [[[self.alfa[t][q]*self.A[q][s]*self.B[s][self.obsTo(self,self.obs[t+1])]*self.beta[t+1][s] for q in range(self.N)]for s in range(self.N)] for t in range(self.T-1)]
			
			for i in range(self.N):			
				for j in range(self.N):
					s2 = 0
					for t in range(self.T-1):
						s2+=ksi[t][i][j]
					self.A[i][j] = s2

			for i in range(self.N):
				s1=0
				for j in range(self.N):
					for t in range(self.T-1):
						s1+=ksi[t][i][j]
				for j in range(self.N):
					self.A[i][j] =self.A[i][j]/s1

			for k in range(self.N):
				for j in range(self.N):
					s2=0
					s1=0
					for i in range(self.N):
						for t in range(self.T-1):
							s2+=ksi[t][i][j]
							if (self.obsTo(self,self.obs[t]) == k):
								s1+=ksi[t][i][j]
						self.B[j][k] = s1/s2

if __name__ == '__main__':
	main()