import math
def GuausB(mu,sig,j,k,x):
	return math.exp(-(x-mu[k][j])**2/(2*sig[k][j]**2))/((2*math.pi)**0.5*sig[k][j])
mu = [[-1,0,1]]
sig = [[0.44,0.44,0.44]]
s=-2
for i in range(70):
	s+=0.1
	print(s)