import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import math
def main():
	f=open('asd.dat','ab')
	for iind in range(4):
		a=np.random.rand(5,5)
		f.write(b'A\r\n')
		np.savetxt(f,a)
	f.close()

if __name__ == '__main__':
	main()