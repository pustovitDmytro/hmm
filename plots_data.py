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

def getcolor(i):
	if i==0:
		return '#cc0099'
	elif i==1:
		return '#3399ff'
	elif i==2:
		return '#99ff00'
	elif i==3:
		return 'grey'
	else:
		return 'silver'

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
y = np.loadtxt('real_data/data2.txt',delimiter='\t')
y = y[::-1]
h= max(y)
l= min(y)
ax1 = plt.subplot(2, 1, 1)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
ax1.plot_date(xdate, y, fmt="b-",label="базовий показник")
plt.xlabel(u'time')
plt.ylabel(u'Медіана валютного курсу')
plt.title(u'Валютний курс в Україні')
state=0
first = True
margins = getMargins(state,y)
for i in margins:
	i = matplotlib.dates.date2num([start+timedelta(days=x) for x in range(i[0], i[1])])
	if first: 
		first = False
		ax1.fill_between(i,min(y),max(y),alpha=0.3,label="Crisis", color='r')
	else: ax1.fill_between(i,min(y),max(y),alpha=0.3, color='r')
plt.legend(loc='upper right',fancybox=True, shadow=True, ncol=6)
ax1 = plt.subplot(2, 1, 2)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
ax1.plot_date(xdate, y, fmt="b-",label="базовий показник")
plt.xlabel(u'time')
plt.ylabel(u'Медіана валютного курсу')
plt.title(u'Валютний курс в Україні')
for state in [0,1,2,3,4]:
	first = True
	margins = getMargins(state,y)
	for i in margins:
		i = matplotlib.dates.date2num([start+timedelta(days=x) for x in range(i[0], i[1])])
		if first: 
			first = False
			ax1.fill_between(i,min(y),max(y),alpha=0.3,label="Δ"+str(state+1), color=getcolor(state))
		else: ax1.fill_between(i,min(y),max(y),alpha=0.3, color=getcolor(state))
plt.legend(loc='upper center',fancybox=True, shadow=True, ncol=6)
plt.show()
plt.clf()

y = np.loadtxt('real_data/reservs893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 1)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'млн.дол.США')
plt.title(u'Золотовалютні резерви')

y = np.loadtxt('real_data/bezr893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 2)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'тис. осіб')
plt.title(u'Показники безробіття')

y = np.loadtxt('real_data/borg893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 3)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'млн. грн')
plt.title(u'Заборгованість за заробітною платою')

y = np.loadtxt('real_data/procent893.data',delimiter='\t')
ax1 = plt.subplot(2, 2, 4)
ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%y"))
ax1.plot_date(xdate, y, fmt="b-")
plt.xlabel(u'time')
plt.ylabel(u'%')
plt.title(u'Облікова ставка НБУ')

plt.show()
