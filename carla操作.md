###carla

####两种模式
- 单独游戏模式 
- 与客户端交互的模式

##### 在两个模式之间切换
- 通过在运行CarlaUE4.sh后面加入 -carla-server命令使其进入
`./CarlaUE4.sh  -carla-server`
- 其他小命令
- 使其启动显示为窗口，分辨率650*350
`./CarlaUE4.sh  -carla-server -windowed -ResX=650 -ResY=350 `
- 使其按照Example.CarlaSettings.ini的内容配置carla
`./CarlaUE4.sh  -carla-settings=Example.CarlaSettings.ini`

#####Example.CarlaSettings.ini
- 可以通过修改 UseNetworking = true 来达到和 -carla-server同样的效果
- 可以通过修改 QualityLevel=Low 使得低配置的电脑运行该程序时更流畅
任何对Example.CarlaSettings.ini的修改都需要通过`./CarlaUE4.sh  -carla-settings=Example.CarlaSettings.ini`生效


####与客户端交互的模式的使用
官方提供了 `driving_benchmark_example.py`,`client_example.py`,`manual_control.py`等客户端用于与carla服务器程序进行通信

- `driving_benchmark_example.py`测试车在不同地方，不同天气条件下的表现。
- `client_example.py`提供自动控制办法和一直向前开车两种模式。 自动控制需要加入  `-a `或者`--autopilot` 启动，还有其他小参数
example：`python client_example.py -a`
- `manual_control.py`提供通过按键来控制小车前进的方法。

- little thing
这些文件可以通过 `-l` 来添加`lidar`,通过`-q=Low`或者`-q=Epic`修改画质

#### 三种不同官方文件之间的共同特性
- 都是通过client 来连接 server
`with make_carla_client(args.host, args.port) as client:`
- 都可以设置类似于Example.CarlaSettings.ini中的参数
```python
	settings = CarlaSettings()
    settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=20,
                    NumberOfPedestrians=40,
                    WeatherId=random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
    settings.randomize_seeds()
```
- 都可以通过settings 设置要添加的 传感器
```
                camera0 = Camera('CameraRGB')
                # Set image resolution in pixels.
                camera0.set_image_size(800, 600)
                # Set its position relative to the car in meters.
                camera0.set_position(0.30, 0, 1.30)
                settings.add_sensor(camera0)

```
- 都可以设置汽车的起点
`client.start_episode(player_start)`
- 数据都是通过`measurements, sensor_data = client.read_data()`得到的
-- 控制命令都是通过类似于下方的代码发送的
```
        control =source/of/control
        control.steer += random.uniform(-0.1, 0.1)
        client.send_control(control)

```

####数据的保存
因为`measurements, sensor_data = client.read_data()`已经获得了所有的数据，只要对这里进行顺序的调整就可了。
下方完成了对`CameraRGB`，`CameraDepth`，`Lidar32`以及`方位，旋转等数据的包装`
```
def record_train_data(measurements,sensor_data):
    # Grey = 0.299 * R + 0.587 * G + 0.114 * B
    rgb_array = np.uint8(0.299 * sensor_data['CameraRGB'].data[:, :, 0] + 0.587 * sensor_data['CameraRGB'].data[:, :, 1] + 0.114 * sensor_data['CameraRGB'].data[:, :, 2])
    seg_array = sensor_data.get('CameraSemSeg', None).data
    # seg_image = sensor_data.get('CameraSemSeg', None)
    # seg_array = image_converter.labels_to_cityscapes_palette(seg_image)
    # seg_array = seg_array[:,:,0]
    player_measurements = measurements.player_measurements
    control = measurements.player_measurements.autopilot_control
    steer = control.steer
    throttle = control.throttle
    brake = control.brake
    hand_brake = control.hand_brake
    reverse = control.reverse
    steer_noise = 0
    gas_noise = 0
    brake_noise = 0
    pos_x = player_measurements.transform.location.x
    pos_y = player_measurements.transform.location.y
    speed = player_measurements.forward_speed * 3.6 # m/s -> km/h
    col_other = player_measurements.collision_other
    col_ped = player_measurements.collision_pedestrians
    col_cars = player_measurements.collision_vehicles
    other_lane = 100 * player_measurements.intersection_otherlane
    offroad = 100 * player_measurements.intersection_offroad
    acc_x = player_measurements.acceleration.x
    acc_y = player_measurements.acceleration.y
    acc_z = player_measurements.acceleration.z
    platform_time = measurements.platform_timestamp
    game_time = measurements.game_timestamp
    orientation_x = player_measurements.transform.orientation.x
    orientation_y = player_measurements.transform.orientation.y
    orientation_z = player_measurements.transform.orientation.z
    command = 2.0
    noise = 0
    camera = 0
    angle = 0
    targets = [steer,throttle,brake,hand_brake,reverse,steer_noise,gas_noise,brake_noise,
               pos_x,pos_y,speed,col_other,col_ped,col_cars,other_lane,offroad,
               acc_x,acc_y,acc_z,platform_time,game_time,orientation_x,orientation_y,orientation_z,
               command,noise,camera,angle]
    return rgb_array,seg_array,targets
```

为什么上述代码可以工作？？
`carla/Util/Proto/carla_server.proto` 文件描述了carla是如何保存measurements数据的。

####设计carla的路线
- 通过设置起点来完成
```

```
