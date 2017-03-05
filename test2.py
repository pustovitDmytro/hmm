s=0
i=0
while True:
	i+=1
	if(i<10):
		s+=1
	elif(i<100):
		s+=2
	elif(i<1000):
		s+=3
	if s==2775:
		break
	print(s)
	print(i)
