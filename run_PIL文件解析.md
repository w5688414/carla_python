###run_CIL.py 解析
###agent是什么（agent是包含了控制命令的类）
- __init__是对网络结构的初始化，同时加载预先训练好的模型
- run_step()返回了control类型的结果，包含steer，throttle，brake
- `agent = ImitationLearning(args.city_name, args.avoid_stopping)`代码段返回了合适的控制数据，而控制应该为`agent.run_step() `
###corl以及CoRL2017是什么（CoRL2017提供了carla环境配置）
- CarlaSettings()控制了车，人，天气，传感器
- Experiment包含CarlaSettings，同时提供了对于起始点，任务量的控制
详见 `corl_2017.py`中的`build_experiments`函数
```
            poses = poses_tasks[iteration]
            vehicles = vehicles_tasks[iteration]
            pedestrians = pedestrians_tasks[iteration]

            conditions = CarlaSettings()
            conditions.set(
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=vehicles,
                NumberOfPedestrians=pedestrians,
                WeatherId=weather
            )
            # Add all the cameras that were set for this experiments

            conditions.add_sensor(camera)

            experiment = Experiment()
            experiment.set(
                Conditions=conditions,
                Poses=poses,
                Task=iteration,
                Repetitions=1
            )
            experiments_vector.append(experiment)
```
- CoRL2017继承了`experiment_suite`，因此初始化的时候就初始化了上面的环境

####self.__planner = Planner(city_name)
- 这个类中包含了路径规划等函数，也就是说包含了地图的信息

####run_driving_benchmark(agent,experiment_suite)
- 重要的步骤
` benchmark.benchmark_agent(experiment_suite, agent, client)`
->
```
self._run_navigation_episode(
                            agent, client, time_out, positions[end_index],
                            str(experiment.Conditions.WeatherId) + '_'
                            + str(experiment.task) + '_' + str(start_index)
                            + '.' + str(end_index))
 ```
->
`_run_navigation_episode`即是按照“得到车的姿态，了解这一步要行走到的目的地（用于控制是否转弯），产生控制命令，发送控制命令”这一过程不断循环进行的