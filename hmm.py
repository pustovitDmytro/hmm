# -*- encoding: utf-8 -*-
import numpy
from abc import ABCMeta, abstractmethod
def main():
	N=10
	obs = numpy.random.binomial(1,0.3,N)
	model = HMM(2,N,obs)
	model.Forward()
	model.BackWard()
	model.Baum_Welch()
	model.show()

class MM():
	__metaclass__ = ABCMeta
	def __init__(self, n,t):
          		self._N = n
          		self._T = t
          		self._A = [[1./n for i in range(self._N)]for j in range(self._N)]
class HMM(MM):
	def __init__(self, n , t, ob):
		super(HMM,self).__init__(n,t)
		self._B = [[1./n for i in range(self._N)]for t in range(self._T)]
		self._obs = ob       
	
	def show(self): 
		print('A:\n',self._A)
		print('B:\n',self._B)
	
	def Forward(self):
		alfa = [[0.5 if j==0 else 0 for i in range(self._N)] for j in range(self._T)]
		for t in range(1,self._T):
			for i in range(self._N):
				for j in range(self._N):
					alfa[t][i]+=alfa[t-1][j]*self._A[i][j]*self._B[t][j]
		self.alfa = alfa

	def BackWard(self):
		beta = [[1 if j==self._T-1 else 0 for i in range(self._N)] for j in range(self._T)]
		for t in range(self._T-1,0,-1):
			for i in range(self._N):
				for j in range(self._N):
					beta[t-1][i]+=beta[t][j]*self._A[i][j]*self._B[t][j]
		self.beta = beta;

	def Baum_Welch(self):
		v = [1,0]
		gama = [[[self.beta[t+1][j]*self._A[i][j]*self._B[t][j]*self.alfa[t][i]for i in range(self._N)] for j in range(self._N)] for t in range(self._T)] 
		A = self._A
		B = self._B
		for i in range(self._N):
			for j in range(self._N):
				s = 0
				Sum = 0;
				for t in range(self._T):
					s+=gama[i][j][t]
				for i1,t1 in zip(range(self._N),range(self._T)):
					Sum+=gama[i1][j][t1]
				A[i][j] = s/Sum;

		for j in range(self._N):
			for t in range(self._T):
				for i1,t1 in zip(range(self._N),range(self._T)):
					Sum+=gama[i1][j][t1]
				for i1,t1 in zip(range(self._N),range(self._T)):
					if self._obs[t1] Sum+=gama[i1][j][t1]




if __name__ == '__main__':
	main()