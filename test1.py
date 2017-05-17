import math
import numpy as np
x = [1,2,3,4,5,6,7,8,9]
#y = list(map(lambda x: math.sin(x),x))
#print (y)
y = [[1,2],[3,4]]
np.savetxt('work.txt',y,delimiter='\t',fmt='%1.8e')
z = np.loadtxt('work.txt',delimiter='\t')
print(z)