# -*- encoding: utf-8 -*-
import numpy
from abc import ABCMeta, abstractmethod
def main():
	#N=10
	#obs = numpy.random.binomial(1,0.3,N)
	
	obs = [2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1]
	
	model = HMM(2,3,obs)
	model.Baum_Welch()
	#model.Baum_Welch()

class MM():
	__metaclass__ = ABCMeta
	def __init__(self,n,t):
          		self.N = n
          		self.T = t
          		self.A = [[.5,.5],[.5,.5]]
          		self.Pi = [.5,.5]
          		#self.A = [[1./n for i in range(self.N)]for j in range(self.N)
class HMM(MM):
	def __init__(self, n , l, ob):
		super(HMM,self).__init__(n,len(ob))
		self.L = l
		#self.B = [[1./n for i in range(self.N)]for t in range(self.L)]
		self.B = [[.4,.1,.5],[0.1,.5,.4]]
		self.obs = ob       
	
	def show(self): 
		print('PI:\n',self.Pi)
		print('A:\n',self.A)
		print('B:\n',self.B)

	def Forward(self):
		alfa = [[self.Pi[i]*self.B[i][self.obs[0]] if t==0 else 0 for i in range(self.N)] for t in range(self.T)]
		for t in range(1,self.T):
			for i in range(self.N):
				for j in range(self.N):
					alfa[t][i]+=alfa[t-1][j]*self.A[j][i]*self.B[i][self.obs[t]]
		self.alfa = alfa

	def BackWard(self):
		beta = [[1 if j==self.T-1 else 0 for i in range(self.N)] for j in range(self.T)]
		for t in range(self.T-1,0,-1):
			for i in range(self.N):
				for j in range(self.N):
					beta[t-1][i]+=beta[t][j]*self.A[i][j]*self.B[j][self.obs[t]]
		self.beta = beta;

	def perevirka(self,ksi,gama):
		print("perevirka:")
		for t in range(self.T-1):
			print("t",t)
			print("gama = ", gama[t])
			print("ksi ",ksi[t])
			
	def Baum_Welch(self):
		for iter in range(20):
			print("Iteration",iter)
			self.show()
			self.Forward()
			self.BackWard()
			#print("alfa: ",self.alfa)
			#print("beta: ",self.beta)
			ksi = [[[self.alfa[t][q]*self.A[q][s]*self.B[s][self.obs[t+1]]*self.beta[t+1][s] for q in range(self.N)]for s in range(self.N)] for t in range(self.T-1)]
			for t in range(self.T-1):
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = self.alfa[t][q]*self.A[q][s]*self.B[s][self.obs[t+1]]*self.beta[t+1][s]

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
						s2+= self.alfa[t][q]*self.A[q][s]*self.B[s][self.obs[t+1]]*self.beta[t+1][s]
				for q in range(self.N):
					for s in range(self.N):
						ksi[t][q][s] = ksi[t][q][s]/s2
				
			#print("s1 = ",s1,"pro= ",pro, "s2 = ",s2)
			#print("gama:\n",gama)
			#print("ksi\n",ksi)
			#self.perevirka(ksi,gama)

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

if __name__ == '__main__':
	main()