def fromFile(name):
	N=3
	f = open(name)
	A = f.readline().split('\t')
	a = []
	for i in A:
		a.append(float(i))
	return a
print(fromFile("data.txt"))