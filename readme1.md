
#Instruction 
在Igasil的基础上做了一些改进和优化；
在场景中增加了多追一的追捕场景；
在算法中增加了我们提出的PGDDPG方法；
增加了一些数据处理的工具；

* 增加设置是否使用reward shaping的控制变量
* 增加设置rs的比例
* 增加设置障碍物数量
* 增加测试胜率的模块
...


#Code Structure
"algorithm" 算法模块
"env" 环境
"result" 我们的训练数据
"tools" 一些数据展示的工具，和我们在试验中用到的猎物

#Have a try
`run_ma_ddpg.py` run ddpg\maddpg

`run_pgddpg_0.5` bate = 0.5

`run_pgddpg_dec.py` bate 递减

`run_pgddpg_vs_fixp.py` 与固定规则对手训练

`show.py` 展示一次训练的render

`test_ones.py` 测试一个prey model模型对抗的成功率

`test_success_rate_all.py` 测试所有与多个prey model对抗的成功率
