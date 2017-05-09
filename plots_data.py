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

ax = plt.subplot(1, 1, 1)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
plt.plot_date(xdate, data, fmt="b-")
plt.show()
