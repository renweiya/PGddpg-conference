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
with open("3v1_/learning_curves/model-prey-s/seed_pgddpg_0.8/pre_trained_prey_20200910204032/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data0 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data0))
with open("3v1_/learning_curves/model-prey-01/seed_pgddpg_0.8/pre_trained_prey_20200910200405/model-prey-01_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data1))
with open("3v1_/learning_curves/model-prey-02/seed_pgddpg_0.8/pre_trained_prey_20200910200419/model-prey-02_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data2))
with open("3v1_/learning_curves/model-prey-03/seed_pgddpg_0.8/pre_trained_prey_20200910200427/model-prey-03_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data3 ))
with open("3v1_/learning_curves/model-prey-04/seed_pgddpg_0.8/pre_trained_prey_20200910200435/model-prey-04_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data4 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data4))
with open("3v1_/learning_curves/model-prey-23/seed_pgddpg_0.8/pre_trained_prey_20200910200115/model-prey-23_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data5 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data5))
with open("3v1_/learning_curves/model-prey-06/seed_pgddpg_0.8/pre_trained_prey_20200910200446/model-prey-06_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data6 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data6))
with open("3v1_/learning_curves/model-prey-07/seed_pgddpg_0.8/pre_trained_prey_20200910200455/model-prey-07_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data7 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data7))
with open("3v1_/learning_curves/model-prey-08/seed_pgddpg_0.8/pre_trained_prey_20200910200504/model-prey-08_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data8 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data8))
with open("3v1_/learning_curves/model-prey-09/seed_pgddpg_0.8/pre_trained_prey_20200910200512/model-prey-09_sucess_record.pkl", 'rb') as fo: 
    pgddpg_dict_data9 = pickle.load(fo, encoding='bytes')
    print(len(pgddpg_dict_data9))

#ddpg
with open("3v1_/learning_curves/model-prey-s/seed_ddpg/20200912103349/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data0 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data0))
with open("3v1_/learning_curves/model-prey-01/seed_ddpg/20200912103401/model-prey-01_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data1))
with open("3v1_/learning_curves/model-prey-02/seed_ddpg/20200912103408/model-prey-02_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data2))
with open("3v1_/learning_curves/model-prey-03/seed_ddpg/20200912103416/model-prey-03_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data3 ))
with open("3v1_/learning_curves/model-prey-04/seed_ddpg/20200912103421/model-prey-04_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data4 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data4))
with open("3v1_/learning_curves/model-prey-23/seed_ddpg/20200912103327/model-prey-23_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data5 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data5))
with open("3v1_/learning_curves/model-prey-06/seed_ddpg/20200912103427/model-prey-06_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data6 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data6))
with open("3v1_/learning_curves/model-prey-07/seed_ddpg/20200912103433/model-prey-07_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data7 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data7))
with open("3v1_/learning_curves/model-prey-08/seed_ddpg/20200912103440/model-prey-08_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data8 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data8))
with open("3v1_/learning_curves/model-prey-09/seed_ddpg/20200912103446/model-prey-09_sucess_record.pkl", 'rb') as fo: 
    ddpg_dict_data9 = pickle.load(fo, encoding='bytes')
    print(len(ddpg_dict_data9))

#maddpg
with open("3v1_/learning_curves/model-prey-s/seed_maddpg/20200910205027/model-prey-s_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data0 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data0))
with open("3v1_/learning_curves/model-prey-01/seed_maddpg/20200910205033/model-prey-01_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data1 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data1))
with open("3v1_/learning_curves/model-prey-02/seed_maddpg/20200910205040/model-prey-02_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data2 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data2))
with open("3v1_/learning_curves/model-prey-03/seed_maddpg/20200910205046/model-prey-03_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data3 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data3 ))
with open("3v1_/learning_curves/model-prey-04/seed_maddpg/20200910205052/model-prey-04_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data4 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data4))
with open("3v1_/learning_curves/model-prey-23/seed_maddpg/20200910205019/model-prey-23_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data5 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data5))
with open("3v1_/learning_curves/model-prey-06/seed_maddpg/20200910205104/model-prey-06_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data6 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data6))
with open("3v1_/learning_curves/model-prey-07/seed_maddpg/20200910205135/model-prey-07_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data7 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data7))
with open("3v1_/learning_curves/model-prey-08/seed_maddpg/20200910205147/model-prey-08_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data8 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data8))
with open("3v1_/learning_curves/model-prey-09/seed_maddpg/20200910205155/model-prey-09_sucess_record.pkl", 'rb') as fo: 
    maddpg_dict_data9 = pickle.load(fo, encoding='bytes')
    print(len(maddpg_dict_data9))



smooth_neighbor=5
start=0
# end=min(len(pgddpg_dict_data0),len(pgddpg_dict_data1),len(pgddpg_dict_data2),len(pgddpg_dict_data3),len(pgddpg_dict_data4),len(pgddpg_dict_data5),len(pgddpg_dict_data6),len(pgddpg_dict_data7),len(pgddpg_dict_data8),len(pgddpg_dict_data9),)
end=400

ddpg_vs_prey00 = savitzky_golay(np.array(ddpg_dict_data0[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey01 = savitzky_golay(np.array(ddpg_dict_data1[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey02 = savitzky_golay(np.array(ddpg_dict_data2[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey03 = savitzky_golay(np.array(ddpg_dict_data3[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey04 = savitzky_golay(np.array(ddpg_dict_data4[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey05 = savitzky_golay(np.array(ddpg_dict_data5[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey06 = savitzky_golay(np.array(ddpg_dict_data6[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey07 = savitzky_golay(np.array(ddpg_dict_data7[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey08 = savitzky_golay(np.array(ddpg_dict_data8[start:end]), smooth_neighbor, 3) 
ddpg_vs_prey09 = savitzky_golay(np.array(ddpg_dict_data9[start:end]), smooth_neighbor, 3) 

maddpg_vs_prey00 = savitzky_golay(np.array(maddpg_dict_data0[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey01 = savitzky_golay(np.array(maddpg_dict_data1[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey02 = savitzky_golay(np.array(maddpg_dict_data2[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey03 = savitzky_golay(np.array(maddpg_dict_data3[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey04 = savitzky_golay(np.array(maddpg_dict_data4[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey05 = savitzky_golay(np.array(maddpg_dict_data5[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey06 = savitzky_golay(np.array(maddpg_dict_data6[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey07 = savitzky_golay(np.array(maddpg_dict_data7[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey08 = savitzky_golay(np.array(maddpg_dict_data8[start:end]), smooth_neighbor, 3) 
maddpg_vs_prey09 = savitzky_golay(np.array(maddpg_dict_data9[start:end]), smooth_neighbor, 3) 

pgddpg_vs_prey00 = savitzky_golay(np.array(pgddpg_dict_data0[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey01 = savitzky_golay(np.array(pgddpg_dict_data1[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey02 = savitzky_golay(np.array(pgddpg_dict_data2[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey03 = savitzky_golay(np.array(pgddpg_dict_data3[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey04 = savitzky_golay(np.array(pgddpg_dict_data4[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey05 = savitzky_golay(np.array(pgddpg_dict_data5[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey06 = savitzky_golay(np.array(pgddpg_dict_data6[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey07 = savitzky_golay(np.array(pgddpg_dict_data7[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey08 = savitzky_golay(np.array(pgddpg_dict_data8[start:end]), smooth_neighbor, 3) 
pgddpg_vs_prey09 = savitzky_golay(np.array(pgddpg_dict_data9[start:end]), smooth_neighbor, 3) 

print(end)

zz = range(0, end-start)
zz=np.multiply(100, zz)
#ax1 = plt.subplot(2,1,1)

plt.figure()

#pgmaddpg
plt.plot(zz, pgddpg_vs_prey00, label='pgddpg_vs_prey00', linewidth=1, linestyle = "dashed",#prey-s
         color='r', marker='o', markerfacecolor='red', markersize=2)
#ddpg
plt.plot(zz, ddpg_vs_prey00, label='ddpg_vs_prey00', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
#maddpg
plt.plot(zz, maddpg_vs_prey00, label='maddpg_vs_prey00', linewidth=1, linestyle = "dashed",#prey-s
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey01, label='pgddpg_vs_prey01', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey02, label='pgddpg_vs_prey02', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey03, label='pgddpg_vs_prey03', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey04, label='pgddpg_vs_prey04', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey05, label='pgddpg_vs_prey05', linewidth=1, linestyle = "dashed",#prey-23
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey06, label='pgddpg_vs_prey06', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey07, label='pgddpg_vs_prey07', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey08, label='pgddpg_vs_prey08', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)
plt.plot(zz, pgddpg_vs_prey09, label='pgddpg_vs_prey09', linewidth=1, linestyle = "dashed",
         color='r', marker='o', markerfacecolor='red', markersize=2)


# plt.tick_params(labelsize=23)


# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 30,
#          }

# plt.title('pgddpg',font2)
# plt.xlabel('iteration',font2)
# plt.ylabel('avg_success_rate',font2)
# plt.legend()
# plt.show()
#ddpg
# plt.plot(zz, ddpg_vs_prey00, label='ddpg_vs_prey00', linewidth=1, linestyle = "dashed",
#          color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey01, label='ddpg_vs_prey01', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey02, label='ddpg_vs_prey02', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey03, label='ddpg_vs_prey03', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey04, label='ddpg_vs_prey04', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey05, label='ddpg_vs_prey05', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey06, label='ddpg_vs_prey06', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey07, label='ddpg_vs_prey07', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey08, label='ddpg_vs_prey08', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)
plt.plot(zz, ddpg_vs_prey09, label='ddpg_vs_prey09', linewidth=1, linestyle = "dashed",
         color='b', marker='v', markerfacecolor='red', markersize=2)


# plt.tick_params(labelsize=23)


# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 30,
#          }

# plt.title('ddpg',font2)
# plt.xlabel('iteration',font2)
# plt.ylabel('avg_success_rate',font2)
# plt.legend()
# plt.show()
#maddpg
# plt.plot(zz, maddpg_vs_prey00, label='maddpg_vs_prey00', linewidth=1, linestyle = "dashed",#prey-s
#          color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey01, label='maddpg_vs_prey01', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey02, label='maddpg_vs_prey02', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey03, label='maddpg_vs_prey03', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey04, label='maddpg_vs_prey04', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey05, label='maddpg_vs_prey05', linewidth=1, linestyle = "dashed",#prey-23
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey06, label='maddpg_vs_prey06', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey07, label='maddpg_vs_prey07', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey08, label='maddpg_vs_prey08', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)
plt.plot(zz, maddpg_vs_prey09, label='maddpg_vs_prey09', linewidth=1, linestyle = "dashed",
         color='g', marker='.', markerfacecolor='red', markersize=2)


# plt.tick_params(labelsize=23)


# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 30,
#          }

# plt.title('maddpg',font2)
# plt.xlabel('iteration',font2)
# plt.ylabel('avg_success_rate',font2)
# plt.legend()
# plt.show()

plt.tick_params(labelsize=23)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

plt.title('Different Seeds',font2)
plt.xlabel('Episodes',font2)
plt.ylabel('avg_success_rate',font2)
plt.legend(labels =[r"pgddpg($\beta=0.8$) vs preys",r"ddpg($\alpha=1$) vs preys",r"maddpg($\alpha=5$) vs preys"])
plt.show()
