import itertools
a = [1,2,3,4]
c = itertools.product(a,repeat=9)
k = 0
for i in c:
	print(k, list(i))
	#d = itertools.p(list(i))
	#for j in d:
		#print(k, j)
	k+=1