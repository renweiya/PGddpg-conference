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
with open("3v1/learning_curves/round_up_base/seed_19_pgddpg_0.8/pre_trained_prey_20200912152603/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data0 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data0))
with open("3v1/learning_curves/round_up_base/seed_238_pgddpg_0.8/pre_trained_prey_20200912152506/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data1))
with open("3v1/learning_curves/round_up_base/seed_454_pgddpg_0.8/pre_trained_prey_20200912152529/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data2))
with open("3v1/learning_curves/round_up_base/seed_476_pgddpg_0.8/pre_trained_prey_20200912152539/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data3 ))
with open("3v1/learning_curves/round_up_base/seed_495_pgddpg_0.8/pre_trained_prey_20200914092116/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data4 = pickle.load(fo, encoding='bytes')   
    print(len(pgddpg_dict_data4))
with open("3v1/learning_curves/round_up_base/seed_530_pgddpg_0.8/pre_trained_prey_20200912152618/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data5 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data5))
with open("3v1/learning_curves/round_up_base/seed_569_pgddpg_0.8/pre_trained_prey_20200912152547/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data6 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data6))
with open("3v1/learning_curves/round_up_base/seed_799_pgddpg_0.8/pre_trained_prey_20200912152438/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data7 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data7))
with open("3v1/learning_curves/round_up_base/seed_917_pgddpg_0.8/pre_trained_prey_20200912152858/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data8 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data8))
with open("3v1/learning_curves/round_up_base/seed_949_pgddpg_0.8/pre_trained_prey_20200912152522/round_up_base_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data9 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data9))

#ddpg
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

    
#maddpg
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_19_maddpg/20200912102742/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data0 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data0))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_238_maddpg/20200912102649/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data1))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_454_maddpg/20200912102707/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data2))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_476_maddpg/20200912102716/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data3 ))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_495_maddpg/20200914093156/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data4 = pickle.load(fo, encoding='bytes') 
    print(len(maddpg_dict_data4))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_530_maddpg/20200912102755/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data5 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data5))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_569_maddpg/20200912102724/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data6 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data6))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_799_maddpg/20200912102621/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo:    #######misss a seed 799
    maddpg_dict_data7 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data7))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_917_maddpg/20200912102733/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data8 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data8))
with open("3v1/learning_curves/round_up_reward_shaping_5/seed_970_maddpg/20200914093227/round_up_reward_shaping_5_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data9 = pickle.load(fo, encoding='bytes')



smooth_neighbor=1
start=0
# end=min(len(pgddpg_dict_data0),len(pgddpg_dict_data1),len(pgddpg_dict_data2),len(pgddpg_dict_data3),len(pgddpg_dict_data4),len(pgddpg_dict_data5),len(pgddpg_dict_data6),len(pgddpg_dict_data7),len(pgddpg_dict_data8),len(pgddpg_dict_data9),)
end=399

pgddpg_seed_0 = savitzky_golay(np.array(pgddpg_dict_data0[start:end]), smooth_neighbor, 3) 
pgddpg_seed_1 = savitzky_golay(np.array(pgddpg_dict_data1[start:end]), smooth_neighbor, 3) 
pgddpg_seed_2 = savitzky_golay(np.array(pgddpg_dict_data2[start:end]), smooth_neighbor, 3) 
pgddpg_seed_3 = savitzky_golay(np.array(pgddpg_dict_data3[start:end]), smooth_neighbor, 3) 
pgddpg_seed_4 = savitzky_golay(np.array(pgddpg_dict_data4[start:end]), smooth_neighbor, 3) 
pgddpg_seed_5 = savitzky_golay(np.array(pgddpg_dict_data5[start:end]), smooth_neighbor, 3) 
pgddpg_seed_6 = savitzky_golay(np.array(pgddpg_dict_data6[start:end]), smooth_neighbor, 3) 
pgddpg_seed_7 = savitzky_golay(np.array(pgddpg_dict_data7[start:end]), smooth_neighbor, 3) 
pgddpg_seed_8 = savitzky_golay(np.array(pgddpg_dict_data8[start:end]), smooth_neighbor, 3) 
pgddpg_seed_9 = savitzky_golay(np.array(pgddpg_dict_data9[start:end]), smooth_neighbor, 3) 

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

maddpg_seed_0 = savitzky_golay(np.array(maddpg_dict_data0[start:end]), smooth_neighbor, 3) 
maddpg_seed_1 = savitzky_golay(np.array(maddpg_dict_data1[start:end]), smooth_neighbor, 3) 
maddpg_seed_2 = savitzky_golay(np.array(maddpg_dict_data2[start:end]), smooth_neighbor, 3) 
maddpg_seed_3 = savitzky_golay(np.array(maddpg_dict_data3[start:end]), smooth_neighbor, 3) 
maddpg_seed_4 = savitzky_golay(np.array(maddpg_dict_data4[start:end]), smooth_neighbor, 3) 
maddpg_seed_5 = savitzky_golay(np.array(maddpg_dict_data5[start:end]), smooth_neighbor, 3) 
maddpg_seed_6 = savitzky_golay(np.array(maddpg_dict_data6[start:end]), smooth_neighbor, 3) 
maddpg_seed_7 = savitzky_golay(np.array(maddpg_dict_data7[start:end]), smooth_neighbor, 3) 
maddpg_seed_8 = savitzky_golay(np.array(maddpg_dict_data8[start:end]), smooth_neighbor, 3) 
maddpg_seed_9 = savitzky_golay(np.array(maddpg_dict_data9[start:end]), smooth_neighbor, 3) 

print(end)

zz = range(0, end-start)
zz=np.multiply(100, zz)
#ax1 = plt.subplot(2,1,1)

plt.figure()
#plt.plot(x,y)
#plt.sca(ax1)
plt.plot(zz, pgddpg_seed_0, label='pgddpg_seed_0', linewidth=1,#prey-s
         color='c', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_1, label='pgddpg_seed_1', linewidth=1,
         color='m', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_2, label='pgddpg_seed_2', linewidth=1,
         color='y', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_3, label='pgddpg_seed_3', linewidth=1,
         color='k', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_4, label='pgddpg_seed_4', linewidth=1,
         color='darkcyan', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_5, label='pgddpg_seed_5', linewidth=1,#prey-23
         color='gray', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_6, label='pgddpg_seed_6', linewidth=1,
         color='darkred', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_7, label='pgddpg_seed_7', linewidth=1,
         color='g', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_8, label='pgddpg_seed_8', linewidth=1,
         color='red', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_seed_9, label='pgddpg_seed_9', linewidth=1,
         color='b', marker='o', markerfacecolor='red', markersize=2)


plt.tick_params(labelsize=23)


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('pgddpg',font2)
plt.xlabel('iteration',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend()
plt.show()

plt.plot(zz, ddpg_seed_0, label='ddpg_seed_0', linewidth=1,
         color='c', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_1, label='ddpg_seed_1', linewidth=1,
         color='m', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_2, label='ddpg_seed_2', linewidth=1,
         color='y', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_3, label='ddpg_seed_3', linewidth=1,
         color='k', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_4, label='ddpg_seed_4', linewidth=1,
         color='darkcyan', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_5, label='ddpg_seed_5', linewidth=1,
         color='gray', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_6, label='ddpg_seed_6', linewidth=1,
         color='darkred', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_7, label='ddpg_seed_7', linewidth=1,
         color='g', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_8, label='ddpg_seed_8', linewidth=1,
         color='red', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_seed_9, label='ddpg_seed_9', linewidth=1,
         color='b', marker='o', markerfacecolor='red', markersize=2)


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

plt.plot(zz, maddpg_seed_0, label='maddpg_seed_0', linewidth=1,#prey-s
         color='c', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_1, label='maddpg_seed_1', linewidth=1,
         color='m', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_2, label='maddpg_seed_2', linewidth=1,
         color='y', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_3, label='maddpg_seed_3', linewidth=1,
         color='k', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_4, label='maddpg_seed_4', linewidth=1,
         color='darkcyan', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_5, label='maddpg_seed_5', linewidth=1,#prey-23
         color='gray', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_6, label='maddpg_seed_6', linewidth=1,
         color='darkred', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_7, label='maddpg_seed_7', linewidth=1,
         color='g', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_8, label='maddpg_seed_8', linewidth=1,
         color='red', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_seed_9, label='maddpg_seed_9', linewidth=1,
         color='b', marker='o', markerfacecolor='red', markersize=2)


plt.tick_params(labelsize=23)


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('maddpg',font2)
plt.xlabel('iteration',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend()
plt.show()
