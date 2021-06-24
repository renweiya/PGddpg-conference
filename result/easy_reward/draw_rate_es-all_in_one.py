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
#success rate    
#ddpg
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_ddpg/20200913233758/round_up_reward_easy_reward_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data1))

#maddpg
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_maddpg/20200913233804/round_up_reward_easy_reward_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data1))
#reward
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_ddpg/20200913233758/round_up_reward_easy_reward_rewards.pkl", 'rb') as fo: 
    dict_data1 = pickle.load(fo, encoding='bytes')  
    print(len(dict_data1[0]))
with open("3v1_easy_reward/learning_curves/round_up_reward_easy_reward/seed_maddpg/20200913233804/round_up_reward_easy_reward_rewards.pkl", 'rb') as fo: 
    dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(dict_data2[0]))



smooth_neighbor=5
start=0
# end=min(len(pgddpg_dict_data0),len(pgddpg_dict_data1),len(pgddpg_dict_data2),len(pgddpg_dict_data3),len(pgddpg_dict_data4),len(pgddpg_dict_data5),len(pgddpg_dict_data6),len(pgddpg_dict_data7),len(pgddpg_dict_data8),len(pgddpg_dict_data9),)
end=400
ddpg_es= savitzky_golay(np.array(ddpg_dict_data1[start:end]), smooth_neighbor, 3) 
maddpg_es= savitzky_golay(np.array(maddpg_dict_data1[start:end]), smooth_neighbor, 3) 

smooth_neighbor_2=500
start_2=0
# end=len(dict_data[0])
end_2=40000
ddpg = savitzky_golay(np.array(np.mean(dict_data1[0:-1],0)[start_2:end_2]), smooth_neighbor_2, 3) 
maddpg = savitzky_golay(np.array(np.mean(dict_data2[0:-1],0)[start_2:end_2]), smooth_neighbor_2, 3) 

print(end)

zz_ori = range(0, end-start)
zz=np.multiply(100, zz_ori)
zz_2 = range(0, end_2-start_2)

fig , ax1 = plt.subplots()
ax2 = ax1.twinx()    # mirror the ax1
# plt.figure()
#plt.plot(x,y)
# plt.sca(ax1)

ax1.plot(zz, ddpg_es, label='Success rate of DDPG (personal reward) ', linewidth=1,linestyle = 'dashed',
         color='b', marker='o',   markersize=3)
ax1.plot(zz, maddpg_es, label='Success rate of MADDPG (personal reward)', linewidth=1,linestyle = 'dashed',
         color='g', marker='v' , markersize=5)

ax2.plot(zz_2, ddpg, label='Average reward of DDPG (personal reward)', linewidth=2,
         color='b')
ax2.plot(zz_2, maddpg, label='Average reward of MADDPG (personal reward)', linewidth=2,
         color='g')


ax1.tick_params(labelsize=23)


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
# plt.xlabel('Episodes')
# plt.ylabel('Avg_Success_Rate',font2)
# plt.legend(prop = font2)


ax1.set_xlabel('Episodes',font2)
ax1.set_ylabel('Success_Rate',font2  )

ax2.set_ylabel('Average-Reward',font2)

#去掉边框
# ax = plt.axes()
# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)



# plt.title('Personal Reward',font2)
ax1.legend(loc="lower center",fontsize = 20)
ax2.legend(loc="upper center",fontsize = 20)

plt.show()
