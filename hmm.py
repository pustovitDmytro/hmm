# -*- encoding: utf-8 -*-
import numpy
from abc import ABCMeta, abstractmethod
def main():
	N=100
	obs = numpy.random.binomial(1,0.3,N)
	model = HMM(2,3)
	model.show()

def Baum_Welch(model):
	#initialization:
	print('')

class MM():
	__metaclass__ = ABCMeta
	def __init__(self, n):
          		self._N = n
          		self._A = numpy.eye(n)
class HMM(MM):
	def __init__(self, n , m):
		super(HMM,self).__init__(n)
		self._B = numpy.eye(m)       
	def show(self): 
		print('A: ',self._A)
		print('B:',self._B)


if __name__ == '__main__':
	main()