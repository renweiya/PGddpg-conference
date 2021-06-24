import pickle
from numpy import *
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

with open("round_up_rewards.pkl", 'rb') as fo: 
    dict_data = pickle.load(fo, encoding='bytes')
  
print(len(dict_data[0]))


smooth_neighbor=500
start=0
end=len(dict_data[0])
#end=170000

dict_data0 = savitzky_golay(np.array(dict_data[0][start:end]), smooth_neighbor, 3) 
dict_data1 = savitzky_golay(np.array(dict_data[1][start:end]), smooth_neighbor, 3) 
dict_data2 = savitzky_golay(np.array(dict_data[2][start:end]), smooth_neighbor, 3) 
dict_data3 = savitzky_golay(np.array(dict_data[3][start:end]), smooth_neighbor, 3) 



zz = range(0, end-start)

ax1 = plt.subplot(4, 2, 1)
ax2 = plt.subplot(4, 2, 2)
ax3 = plt.subplot(4, 2, 3)
ax4 = plt.subplot(4, 2, 4)

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.sca(ax1)
plt.plot(zz, dict_data0, label='predator-0-reward', linewidth=1,
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.xlabel('episodes',font2)
plt.ylabel('predator-0-reward',font2)

plt.sca(ax2)
plt.plot(zz, dict_data1, label='predator-1-reward', linewidth=1,
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.xlabel('episodes',font2)
plt.ylabel('predator-1-reward',font2)

plt.sca(ax3)
plt.plot(zz, dict_data2, label='predator-2-reward', linewidth=1,
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.xlabel('episodes',font2)
plt.ylabel('predator-2-reward',font2)

plt.sca(ax4)
plt.plot(zz, dict_data3, label='prey-reward', linewidth=1,
         color='b', marker='o', markerfacecolor='blue', markersize=2)
plt.xlabel('episodes',font2)
plt.ylabel('prey-reward',font2)
plt.show()