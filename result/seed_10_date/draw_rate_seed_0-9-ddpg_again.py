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

with open("3v1/learning_curves/round_up_reward_shaping/seed_19_ddpg/20200912102551/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data0 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data0))
with open("3v1/learning_curves/round_up_reward_shaping/seed_238_ddpg/20200912102354/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data1))
with open("3v1/learning_curves/round_up_reward_shaping/seed_454_ddpg/20200912102455/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data2))
with open("3v1/learning_curves/round_up_reward_shaping/seed_476_ddpg/20200912102512/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data3 ))
with open("3v1/learning_curves/round_up_reward_shaping/seed_495_ddpg/20200912102411/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data4 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data4))
with open("3v1/learning_curves/round_up_reward_shaping/seed_530_ddpg/20200912102307/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data5 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data5))
with open("3v1/learning_curves/round_up_reward_shaping/seed_569_ddpg/20200912102527/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data6 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data6))
with open("3v1/learning_curves/round_up_reward_shaping/seed_799_ddpg/20200912102334/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data7 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data7))
with open("3v1/learning_curves/round_up_reward_shaping/seed_917_ddpg/20200912102540/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data8 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data8))
with open("3v1/learning_curves/round_up_reward_shaping/seed_949_ddpg/20200912102425/round_up_reward_shaping_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data9 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data9))



smooth_neighbor=1
start=0
# end=min(len(ddpg_dict_data0),len(ddpg_dict_data1),len(ddpg_dict_data2),len(ddpg_dict_data3),len(ddpg_dict_data4),len(ddpg_dict_data5),len(ddpg_dict_data6),len(ddpg_dict_data7),len(ddpg_dict_data8),len(ddpg_dict_data9),)
end=400

ddpg_seed_0 = savitzky_golay(np.array(ddpg_dict_data0[start:end]), smooth_neighbor, 3) 
ddpg_seed_1 = savitzky_golay(np.array(ddpg_dict_data1[start:end]), smooth_neighbor, 3) 
ddpg_seed_2 = savitzky_golay(np.array(ddpg_dict_data2[start:end]), smooth_neighbor, 3) 
ddpg_seed_3 = savitzky_golay(np.array(ddpg_dict_data3[start:end]), smooth_neighbor, 3) 
ddpg_seed_4 = savitzky_golay(np.array(ddpg_dict_data4[start:end]), smooth_neighbor, 3) 
ddpg_seed_5 = savitzky_golay(np.array(ddpg_dict_data5[start:end]), smooth_neighbor, 3) 
ddpg_seed_6 = savitzky_golay(np.array(ddpg_dict_data6[start:end]), smooth_neighbor, 3) 
ddpg_seed_7 = savitzky_golay(np.array(ddpg_dict_data7[start:end]), smooth_neighbor, 3) 
ddpg_seed_8 = savitzky_golay(np.array(ddpg_dict_data8[start:end]), smooth_neighbor, 3) 
ddpg_seed_9 = savitzky_golay(np.array(ddpg_dict_data9[start:end]), smooth_neighbor, 3) 

print(end)



zz = range(0, end-start)
zz=np.multiply(100, zz)
#ax1 = plt.subplot(2,1,1)

plt.figure()
#plt.plot(x,y)
#plt.sca(ax1)
seed = np.array([[ddpg_seed_0[i],ddpg_seed_1[i],ddpg_seed_2[i],ddpg_seed_3[i],ddpg_seed_4[i],ddpg_seed_5[i],ddpg_seed_6[i],ddpg_seed_7[i],ddpg_seed_8[i],ddpg_seed_9[i]] for i in range(start,end) ])
seed_max = np.max(seed,1)
seed_min = np.min(seed,1)
seed_mean = np.mean(seed,1) 
plt.plot(zz, seed_mean, label='ddpg_seed_0', linewidth=1,color='c' )
plt.fill_between(zz, seed_min, seed_max,color='c',alpha = 0.2)


plt.tick_params(labelsize=23)

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('ddpg',font2)
plt.xlabel('iteration',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend()
plt.show()
