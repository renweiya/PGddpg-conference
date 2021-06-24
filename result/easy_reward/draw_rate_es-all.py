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
#pgddpg
# with open("3v1_p_speed-03/learning_curves/model-prey-s/seed_pgddpg_0.8/pre_trained_prey_20200910222526/model-prey-s_sucess_record.pkl", 'rb') as fo: 
#     pgddpg_dict_data1 = pickle.load(fo, encoding='bytes')
#     print(len(pgddpg_dict_data1))
    
#ddpg
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_ddpg/20200913233758/round_up_reward_easy_reward_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data1))

#maddpg
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_maddpg/20200913233804/round_up_reward_easy_reward_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data1))


smooth_neighbor=5
start=0
# end=min(len(pgddpg_dict_data0),len(pgddpg_dict_data1),len(pgddpg_dict_data2),len(pgddpg_dict_data3),len(pgddpg_dict_data4),len(pgddpg_dict_data5),len(pgddpg_dict_data6),len(pgddpg_dict_data7),len(pgddpg_dict_data8),len(pgddpg_dict_data9),)
end=400

ddpg_es= savitzky_golay(np.array(ddpg_dict_data1[start:end]), smooth_neighbor, 3) 
maddpg_es= savitzky_golay(np.array(maddpg_dict_data1[start:end]), smooth_neighbor, 3) 
# pgddpg(easy reward) = savitzky_golay(np.array(pgddpg_dict_data1[start:end]), smooth_neighbor, 3) 

print(end)

zz = range(0, end-start)
zz=np.multiply(100, zz)
#ax1 = plt.subplot(2,1,1)

plt.figure()
#plt.plot(x,y)
#plt.sca(ax1)
# plt.plot(zz, pgddpg_vs_prey01, label='pgddpg_vs_prey(max_speed = 0.3)', linewidth=1,
#          color='R', marker='x',   markersize=5)
plt.plot(zz, ddpg_es, label='DDPG (personal reward) ', linewidth=1,
         color='b', marker='o',   markersize=3)
plt.plot(zz, maddpg_es, label='MADDPG (personal reward)', linewidth=1,
         color='g', marker='v' , markersize=5)


plt.tick_params(labelsize=23)


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

# plt.title('Easy Reward',font2)
plt.xlabel('Episodes',font2)
plt.ylabel('Avg_Success_Rate',font2)
plt.legend(prop = font2)
#去掉边框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
