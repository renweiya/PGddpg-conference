# #######删除当前文件夹下重复的模型#######
# Eg：ls | grep -v  checkpoint |grep -v 40000|xargs rm  在包含“checkpoint”文件的文件夹下，删除除了“checkpoint”和包含“40000”的其他文件
# Note：”/media/ps/data/pgddpg-twice/exp_result/PGddpg-sum/result/test_all/test_model“需要修改为当前绝对路径
# Author： JoneY 2021.6.23

aa=$(find ./ -name "checkpoint" | sed 's/.checkpoint$//g')
for ii in ${aa}
do
    cd /media/ps/data/pgddpg-twice/exp_result/PGddpg-sum/result/test_all/test_model
    cd $ii
    echo $ii
    #ls | grep -v  checkpoint | grep -v 38000 | grep -v 39000|grep -v 40000
    ls | grep -v  checkpoint |grep -v 40000|xargs rm

done
