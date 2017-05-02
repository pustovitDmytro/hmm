import numpy
from abc import ABCMeta, abstractmethod
def main():

def fromFile(name):
	f = open(name)
	A = f.readline().split('\t')
	a = []
	for i in A:
		a.append(float(i))
	return a

class MM():
	__metaclass__ = ABCMeta
	def __init__(self,n,a,pi):
          		self.N = n		
          		self.A = a
          		self.Pi = pi
class HMM(MM):
	def __init__(self, n , l, ob, a, pi, b):
		super(HMM,self).__init__(n,a,pi)
		self.L = l
		self.B = b
		self.T = len(ob)
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

	def Baum_Welch(self):
		for iter in range(500):
			self.Forward()
			self.BackWard()
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