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

with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_ddpg/20200913233758/round_up_reward_easy_reward_rewards.pkl", 'rb') as fo: 
    dict_data1 = pickle.load(fo, encoding='bytes')  
    print(len(dict_data1[0]))
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_maddpg/20200913233804/round_up_reward_easy_reward_rewards.pkl", 'rb') as fo: 
    dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(dict_data2[0]))


smooth_neighbor=500
start=0
# end=len(dict_data[0])
end=40000

ddpg = savitzky_golay(np.array(np.mean(dict_data1[0:-1],0)[start:end]), smooth_neighbor, 3) 
maddpg = savitzky_golay(np.array(np.mean(dict_data2[0:-1],0)[start:end]), smooth_neighbor, 3) 
# dict_data1 = savitzky_golay(np.array(dict_data[1][start:end]), smooth_neighbor, 3) 
# dict_data2 = savitzky_golay(np.array(dict_data[2][start:end]), smooth_neighbor, 3) 
# dict_data3 = savitzky_golay(np.array(dict_data[3][start:end]), smooth_neighbor, 3) 



zz = range(0, end-start)


plt.figure()

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.plot(zz, ddpg, label='DDPG (personal reward)', linewidth=1,
         color='b', marker='.', markerfacecolor='blue', markersize=2)
plt.plot(zz, maddpg, label='MADDPG (personal reward)', linewidth=1,
         color='g', marker='.', markerfacecolor='blue', markersize=2)

plt.xlabel('Episodes',font2)
plt.ylabel('Average-Reward',font2)
# plt.title("Easy Reward",font2)
plt.legend(prop = font2)
#去掉边框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
