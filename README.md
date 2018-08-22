## 此代码的作用

[carla(一款自动驾驶测试虚拟环境](www.carla.org)中提供了如何利用carla官方代码实现一些简单的功能。


该代码提供了更为详细的功能
- 收集数据并保存 `debugsaveimg.py`
- 使用XBOX游戏手柄对赛车进行控制 `manual_record.py`
- testfunction中提供了众多用于分析数据的代码.`read.py` `plotoriginal.py` `plot.py`


##manual_record.py
在打开了carla服务器端并开启其通信功能之后，运行`python3 manual_record.py`可以实现如下功能:
1.使用XBOX来操作赛车的行动
2.在操作的同时记录相关数据
3.可以通过调整代码来实现使用 键盘来操作，或者不记录数据的功能，可以改变天气，录制时间，录制画质等等
4.请关注代码的运行提示

##testfunction中的代码
`plotoriginal.py`中提供了分析 保存steer，gas，brake分布情况的代码。
`plot`用于与`plotoriginal.py`配合（利用pk文件），将图画出来
`read.py`用于分析采集文件中的图片的实际样子，是否合理
testfunction中的其他文件都不成熟，可以不看

