from datetime import date
import statistics
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np

def compute(arr,c):
	res = [0 for i in range(c)]
	for i in range(c,len(arr)):
		res.append(arr[i]-arr[i-c])
	return res

def yinstate(obs,y):
	mean = statistics.mean(y)
	stdev = statistics.stdev(y)
	if obs>mean+stdev:
		return 0
	elif obs>mean+0.5*stdev:
		return 1
	elif obs>mean-0.5*stdev:
		return 2
	elif obs>mean-stdev:
		return 3
	else:
		return 4

def getMargins(state, y):
	a = []
	res = []
	y = list(map(lambda x: yinstate(x,y)==state,y))
	for i in range (1,len(y)):
		if y[i]:
			if i==1: a.append(i)
			elif y[i-1]==False: a.append(i)
			if i==len(y)-1 or y[i+1]==False:
				a.append(i)
				res.append(a)
				a=[]
	return res

#base = datetime.datetime.today()
end = date(2017,4,3)
start = date(2014,10,24)
numdays = (end-start).days+1;
print(numdays)
dateRange = [start+timedelta(days=x) for x in range(0, numdays)]
xdate = matplotlib.dates.date2num(dateRange)
y = np.loadtxt('data2.txt',delimiter='\t')
y = y[::-1]
h= max(y)
l= min(y)
ax1 = plt.subplot(2, 1, 1)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'Медіана валютного курсу')
plt.title(u'Валютний курс в Україні')
first = True
margins = getMargins(0,y)
for i in margins:
	i = matplotlib.dates.date2num([start+timedelta(days=x) for x in range(i[0], i[1])])
	if first: 
		first = False
		plt.fill_between(i,min(y),max(y),alpha=0.3,label="Crisis", color='r')
	else: plt.fill_between(i,min(y),max(y),alpha=0.3, color='r')
plt.legend(loc='upper right',fancybox=True, shadow=True, ncol=3)
#plt.show()

y = np.loadtxt('reservs893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 1)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'млн.дол.США')
plt.title(u'Золотовалютні резерви')

y = np.loadtxt('bezr893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 2)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'тис. осіб')
plt.title(u'Показники безробіття')

y = np.loadtxt('borg893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 3)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'млн. грн')
plt.title(u'Заборгованість за заробітною платою')

y = np.loadtxt('procent893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 4)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'%')
plt.title(u'Облікова ставка НБУ')

plt.show()
