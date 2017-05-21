import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import math
import hmm
def main():
	N=4
	M = 2
	x_i = range(N)
	x  = np.random.binomial(3,0.3,N)
	mu = [0,1,0,1,0]
	sig = [0.4,0.4,0.4,0.4]
	print(x)
	print(mu)
	print(sig)

if __name__ == '__main__':
	main()