# -*- encoding: utf-8 -*-
import numpy
from abc import ABCMeta, abstractmethod
def main():
	N=10
	obs = numpy.random.binomial(1,0.3,N)
	model = HMM(2,2,obs)
	model.Baum_Welch()
	model.show()

class MM():
	__metaclass__ = ABCMeta
	def __init__(self, n,t):
          		self.N = n
          		self.T = t
          		self.A = [[1./n for i in range(self.N)]for j in range(self.N)]
class HMM(MM):
	def __init__(self, n , l, ob):
		super(HMM,self).__init__(n,len(ob))
		self.L = l
		self.B = [[1./n for i in range(self.N)]for t in range(self.L)]
		self.obs = ob       
	
	def show(self): 
		print('A:\n',self.A)
		print('B:\n',self.B)

	def Forward(self):
		alfa = [[0.5 if j==0 else 0 for i in range(self.N)] for j in range(self.T)]
		for t in range(1,self.T):
			for i in range(self.N):
				for j in range(self.N):
					alfa[t][i]+=alfa[t-1][j]*self.A[i][j]*self.B[self.obs[t]][j]
		self.alfa = alfa

	def BackWard(self):
		beta = [[1 if j==self.T-1 else 0 for i in range(self.N)] for j in range(self.T)]
		for t in range(self.T-1,0,-1):
			for i in range(self.N):
				for j in range(self.N):
					beta[t-1][i]+=beta[t][j]*self.A[i][j]*self.B[self.obs[t]][j]
		self.beta = beta;

	def Baum_Welch(self):
		
		for iter in range(10):
			self.Forward()
			self.BackWard()
			gama = [[[self.beta[t+1][j]*self.A[i][j]*self.B[self.obs[t]][j]*self.alfa[t][i]for i in range(self.N)] for j in range(self.N)] for t in range(self.T)]
			for i in range(self.N):
				for j in range(self.N):
					s = 0
					Sum = 0;
					for t in range(self.T):
						s+=gama[i][j][t]
					for i1,t1 in zip(range(self.N),range(self.T)):
						Sum+=gama[i][i1][t1]
					self.A[i][j] = s/Sum
	
			for j in range(self.N):
				for k in range(self.L):
					Sum=0
					s =0
					for i1,t1 in zip(range(self.N),range(self.T)):
						Sum+=gama[i1][j][t1]
					for i1,t1 in zip(range(self.N),range(self.T)):
						if self.obs[t1]==k: s+=gama[i1][j][t1]
					self.B[j][k]=s/Sum
			print("Iteration",iter)
			self.show()

if __name__ == '__main__':
	main()