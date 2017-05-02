import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
allMonth = MonthLocator()
alldays = DayLocator()
monthFormatter = DateFormatter('%b %y')
f = open('data2.txt', 'r')
file = f.read()
data = file.split('\n')
data = list(map(float,data))
def compute(arr,c):
	res = [0 for i in range(c)]
	for i in range(c,len(arr)):
		res.append(arr[i]-arr[i-c])
	return res
data=data[::-1]
data2 = compute(data,5);
#x=range(893)
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(allMonth)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(monthFormatter)
ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.plot(data2)
plt.show()