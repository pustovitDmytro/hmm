import os
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates

#base = datetime.datetime.today()
base = date(2017,4,3)
numdays = 893;
dateRange = [base - timedelta(days=x) for x in range(0, numdays)]
xdate = matplotlib.dates.date2num(dateRange)
intresting = [date(2015,2,1) + timedelta(days=x) for x in range(0, 30)]

f = open('data2.txt', 'r')
file = f.read()
data = file.split('\n')
data = list(map(float,data))
def compute(arr,c):
	res = [0 for i in range(c)]
	for i in range(c,len(arr)):
		res.append(arr[i]-arr[i-c])
	return res
#data=data[::-1]
data2 = compute(data,5)
x=range(893)
x2date = matplotlib.dates.date2num(intresting)
#h= [max(data) for x in range(len(x2date))]
#l= [min(data) for x in range(len(x2date))]
h= max(data)
l= min(data)
ax = plt.subplot(1, 1, 1)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
plt.fill_between(x2date,h,l ,alpha=0.3, color='red')
plt.plot_date(xdate, data, fmt="b-")
#plt.plot_date(xdate, [1.5 for x in range(len(xdate))], fmt="b-")
plt.show()
