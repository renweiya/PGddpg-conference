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
with open("3v1_p_speed-03/learning_curves/model-prey-s/seed_pgddpg_0.8/pre_trained_prey_20200910222526/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data1))
with open("3v1_p_speed-05/learning_curves/model-prey-s/seed_pgddpg_0.8/pre_trained_prey_20200910222450/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data2))
with open("3v1_p_speed-07/model-prey-s/seed_pgddpg_0.8/pre_trained_prey_20200910204032/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data3 ))
    
#ddpg
with open("3v1_p_speed-03/learning_curves/model-prey-s/seed_ddpg/20200912153644/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data1))
with open("3v1_p_speed-05/learning_curves/model-prey-s/seed_ddpg/20200912153750/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data2))
with open("3v1_p_speed-07/model-prey-s/seed_ddpg/20200912103349/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data3 ))

#maddpg
with open("3v1_p_speed-03/learning_curves/model-prey-s/seed_maddpg/20200912153708/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data1))
with open("3v1_p_speed-05/learning_curves/model-prey-s/seed_maddpg/20200912153812/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data2))
with open("3v1_p_speed-07/model-prey-s/seed_maddpg/20200910205027/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data3 ))


smooth_neighbor=5
start=0
# end=min(len(pgddpg_dict_data0),len(pgddpg_dict_data1),len(pgddpg_dict_data2),len(pgddpg_dict_data3),len(pgddpg_dict_data4),len(pgddpg_dict_data5),len(pgddpg_dict_data6),len(pgddpg_dict_data7),len(pgddpg_dict_data8),len(pgddpg_dict_data9),)
end=400

ddpg_vs_prey01 = savitzky_golay(np.array(ddpg_dict_data1[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey02 = savitzky_golay(np.array(ddpg_dict_data2[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey03 = savitzky_golay(np.array(ddpg_dict_data3[start:end]), smooth_neighbor, 3) 

maddpg_vs_prey01 = savitzky_golay(np.array(maddpg_dict_data1[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey02 = savitzky_golay(np.array(maddpg_dict_data2[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey03 = savitzky_golay(np.array(maddpg_dict_data3[start:end]), smooth_neighbor, 3) 

pgddpg_vs_prey01 = savitzky_golay(np.array(pgddpg_dict_data1[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey02 = savitzky_golay(np.array(pgddpg_dict_data2[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey03 = savitzky_golay(np.array(pgddpg_dict_data3[start:end]), smooth_neighbor, 3) 

print(end)

zz = range(0, end-start)
zz=np.multiply(100, zz)
#ax1 = plt.subplot(2,1,1)

plt.figure()
#plt.plot(x,y)
#plt.sca(ax1)
plt.plot(zz, pgddpg_vs_prey01, label='pgddpg_vs_prey(max_speed = 0.3)', linewidth=1, 
         color='R', marker='s', markersize=4 ,)
plt.plot(zz, ddpg_vs_prey01, label='ddpg(rs)_vs_prey(max_speed = 0.3)', linewidth=1, 
         color='b', marker='o', markersize=4 ,)
#maddpg
plt.plot(zz, maddpg_vs_prey01, label='maddpg(rs)_vs_prey(max_speed = 0.3)', linewidth=1, 
         color='g', marker='v', markersize=4 ,)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('Prey-00 With Speed 0.3',font2)
plt.xlabel('Episodes',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend()
plt.show()
plt.plot(zz, pgddpg_vs_prey02, label='pgddpg_vs_prey(max_speed = 0.5)', linewidth=1, 
         color='r', marker='s', markersize=4 ,)
plt.plot(zz, ddpg_vs_prey02, label='ddpg(rs)_vs_prey(max_speed = 0.5)', linewidth=1, 
         color='b', marker='o', markersize=4 ,)
plt.plot(zz, maddpg_vs_prey02, label='maddpg(rs)_vs_prey(max_speed = 0.5)', linewidth=1, 
         color='g', marker='v', markersize=4 ,)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('Prey-00 With  Speed 0.5',font2)
plt.xlabel('Episodes',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend()
plt.show()
plt.plot(zz, pgddpg_vs_prey03, label='pgddpg_vs_prey(max_speed = 0.7)', linewidth=1, 
         color='r', marker='s', markersize=4 ,)
plt.plot(zz, ddpg_vs_prey03, label='ddpg(rs)_vs_prey(max_speed = 0.7)', linewidth=1, 
         color='b', marker='o', markersize=4 ,)
plt.plot(zz, maddpg_vs_prey03, label='maddpg(rs)_vs_prey(max_speed = 0.7)', linewidth=1, 
         color='g', marker='v', markersize=4 ,)


plt.tick_params(labelsize=23)


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('Prey-00 With Speed 0.7',font2)
plt.xlabel('Episodes',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend()
plt.show()
