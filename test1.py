import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import math
def goglobal(self,obs):
	t = 10 - int(obs*10-12)
	return t
def main():
	s=1.1
	for i in range(15):
		s+=0.1
		print(s,goglobal(0,s))

if __name__ == '__main__':
	main()