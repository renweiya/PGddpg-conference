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



smooth_neighbor=5
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
seed = np.array([[pgddpg_seed_0[i],pgddpg_seed_1[i],pgddpg_seed_2[i],pgddpg_seed_3[i],pgddpg_seed_4[i],pgddpg_seed_5[i],pgddpg_seed_6[i],pgddpg_seed_7[i],pgddpg_seed_8[i],pgddpg_seed_9[i]] for i in range(start,end) ])
seed_max = np.max(seed,1)
seed_min = np.min(seed,1)
seed_mean = np.mean(seed,1) 
plt.plot(zz, seed_mean, label='pgddpg_seed', linewidth=1,color='r' )
plt.fill_between(zz, seed_min, seed_max,color='r',alpha = 0.1)

seed = np.array([[ddpg_seed_0[i],ddpg_seed_1[i],ddpg_seed_2[i],ddpg_seed_3[i],ddpg_seed_4[i],ddpg_seed_5[i],ddpg_seed_6[i],ddpg_seed_7[i],ddpg_seed_8[i],ddpg_seed_9[i]] for i in range(start,end) ])
seed_max = np.max(seed,1)
seed_min = np.min(seed,1)
seed_mean = np.mean(seed,1) 
plt.plot(zz, seed_mean, label='ddpg_seed', linewidth=1,color='b' )
plt.fill_between(zz, seed_min, seed_max,color='b',alpha = 0.1)


seed = np.array([[maddpg_seed_0[i],maddpg_seed_1[i],maddpg_seed_2[i],maddpg_seed_3[i],maddpg_seed_4[i],maddpg_seed_5[i],maddpg_seed_6[i],maddpg_seed_7[i],maddpg_seed_8[i],maddpg_seed_9[i]] for i in range(start,end) ])
seed_max = np.max(seed,1)
seed_min = np.min(seed,1)
seed_mean = np.mean(seed,1) 
plt.plot(zz, seed_mean, label='maddpg_seed', linewidth=1,color='g' )
plt.fill_between(zz, seed_min, seed_max,color='g',alpha = 0.1)
# plt.tick_params(labelsize=23) 
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

# plt.title('Different Seed',font2)
plt.xlabel('Episodes',font2)
plt.ylabel('Avg_Success_Rate',font2)
plt.legend( labels = [r"PGDDPG ($\beta=0.8$) vs. prey-00",r"DDPG ($\alpha=1$) vs. prey-00",r"MADDPG ($\alpha=5$) vs. prey-00"],fontsize=20)
#去掉边框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()