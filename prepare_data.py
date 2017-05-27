from datetime import date
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates
import csv
import numpy as np

def prepare(input,output,format = "%d.%m.%Y"):
	y=[]
	with open(input, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=';')
		for row in spamreader:
			start = datetime.datetime.strptime(row[0],format)
			end = datetime.datetime.strptime(row[1],format)
			value  = row[2]
			for i in range((end-start).days+1):
				y.append(float(value))
	np.savetxt(output,y,delimiter='\t',fmt='%1.8e')
	print(len(y))
def singleprepare(input,output,format = "%d.%m.%Y", prev="24.10.2014"):
	y=[]
	with open(input, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=';')
		start = datetime.datetime.strptime(prev,format)
		for row in spamreader:
			end = datetime.datetime.strptime(row[0],format)
			value  = row[1]
			for i in range((end-start).days+1):
				y.append(float(value))
			start = end + timedelta(days=1)
	np.savetxt(output,y,delimiter='\t',fmt='%1.8e')
	print(len(y))


start = date(2014,10,24)
end = date(2017,4,3)
print(end-start+ timedelta(days=1))
singleprepare("raw_data/bezrobitta.mine","bezr893.data")




