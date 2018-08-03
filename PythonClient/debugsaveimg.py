#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla import image_converter

import h5py
import numpy as np
import os
import sys
import termios

def mkdir(path):
    path = path.strip()
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return True

def fileCounter(path):
    ls = os.listdir(path)
    return len(ls)

def press_any_key_exit(msg):
    # 获取标准输入的描述符
    fd = sys.stdin.fileno()
    # 获取标准输入(终端)的设置
    old_ttyinfo = termios.tcgetattr(fd)
    # 配置终端
    new_ttyinfo = old_ttyinfo[:]
    # 使用非规范模式(索引3是c_lflag 也就是本地模式)
    new_ttyinfo[3] &= ~termios.ICANON
    # 关闭回显(输入不会被显示)
    new_ttyinfo[3] &= ~termios.ECHO
    # 输出信息
    sys.stdout.write(msg)
    sys.stdout.flush()
    # 使设置生效
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    # 从终端读取
    os.read(fd, 7)
    # 还原终端设置
    termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

def record_train_data(measurements,sensor_data):

    ## Collect sensordata->sensors
    # Grey = 0.299 * R + 0.587 * G + 0.114 * B
    rgb_array = np.uint8(0.299 * sensor_data['CameraRGB'].data[:, :, 0] + 0.587 * sensor_data['CameraRGB'].data[:, :, 1] + 0.114 * sensor_data['CameraRGB'].data[:, :, 2])
    seg_array = sensor_data.get('CameraSemSeg', None).data
    depth_array = sensor_data.get('CameraDepth', None)
    depth_array = image_converter.depth_to_logarithmic_grayscale(depth_array)
    depth_array = depth_array[:,:,0]
    lidar_measurement = sensor_data.get('Lidar32', None)
    lidar_data = np.array(lidar_measurement.data[:, :2])
    lidar_data *= 2.0
    lidar_data += 100.0
    lidar_data = np.fabs(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    # draw lidar
    lidar_img_size = (200, 200)
    lidar_img = np.zeros(lidar_img_size)
    lidar_img[tuple(lidar_data.T)] = 255
    sensors = {'CameraRGB':rgb_array, 'CameraSemSeg':seg_array, 'CameraDepth':depth_array,'Lidar32':lidar_img}



    ## collect measurementdata->targets
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
    return sensors,targets


def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 5
    cut_per_episode = 40
    frames_per_cut = 200


    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
        for episode in range(0, number_of_episodes):
            print("input any key to continue...")
            start = input()
            # each episode dir store a set of traindata.  if dir not existed, then make it
            pathdir = '/home/kadn/dataTrain/episode_{:0>3}'.format(episode)
            mkdir(pathdir)

            # Create a CarlaSettings object. This object is a wrapper around
            # the CarlaSettings.ini file. Here we set the configuration we
            # want for the new episode.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=20,
                NumberOfPedestrians=40,
                # WeatherId=random.choice([1, 3, 7, 8, 14]),
                WeatherId = 1,
                QualityLevel=args.quality_level)
            settings.randomize_seeds()
            # Now we want to add a couple of cameras to the player vehicle.
            # We will collect the images produced by these cameras every
            # frame.

            # The default camera captures RGB images of the scene.
            camera0 = Camera('CameraRGB')
            # Set image resolution in pixels.
            camera0.set_image_size(88, 200)
            # Set its position relative to the car in meters.
            camera0.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera0)

            # Let's add another camera producing ground-truth depth.
            camera1 = Camera('CameraDepth', PostProcessing='Depth')
            camera1.set_image_size(200, 88)
            camera1.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera1)

            camera2 = Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
            camera2.set_image_size(88, 200)
            camera2.set_position(2.0, 0.0, 1.4)
            camera2.set_rotation(0.0, 0.0, 0.0)
            settings.add_sensor(camera2)
            # if args.lidar:
            lidar = Lidar('Lidar32')
            lidar.set_position(0, 0, 2.50)
            lidar.set_rotation(0, 0, 0)
            lidar.set(
                Channels=0,
                Range=30,
                PointsPerSecond=200000,
                RotationFrequency=10,
                UpperFovLimit=0,
                LowerFovLimit=0)
            settings.add_sensor(lidar)

            # else:
            #
            #     # Alternatively, we can load these settings from a file.
            #     with open(args.settings_filepath, 'r') as fp:
            #         settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            # number_of_player_starts = len(scene.player_start_spots)
            # player_start = random.randint(0, max(0, number_of_player_starts - 1))
            player_start = 1

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            # Start a new episode.
            client.start_episode(player_start)

            for cut_per_episode in range(0,cut_per_episode):
                num = fileCounter(pathdir)
                filename = "data_{:0>6}.h5".format(num)
                filepath = pathdir + '/' + filename
                f = h5py.File(filepath, "w")
                rgb_file = f.create_dataset("rgb", (200, 88, 200), np.uint8)
                seg_file = f.create_dataset("seg", (200, 88, 200), np.uint8)
                lidar_file = f.create_dataset('lidar',(200,200,200),np.uint8)
                startendpoint = f.create_dataset('startend',(1,2),np.float32)
                targets_file = f.create_dataset("targets", (200, 28), np.float32)
                index_file = 0

                # Iterate every frame in the episode.
                for frame in range(0, frames_per_cut):
                    # Read the data produced by the server this frame.
                    measurements, sensor_data = client.read_data()

                    # get data and store
                    sensors, targets_data = record_train_data(measurements,sensor_data)
                    rgb_file[:,:,index_file] = sensors['rgb']
                    seg_file[:,:,index_file] = sensors['seg']
                    lidar_file[:,:,index_file] = sensors['lidar']
                    targets_file[index_file,:] = targets_data
                    index_file += 1

                    # We can access the encoded data of a given image as numpy
                    # array using its "data" property. For instance, to get the
                    # depth value (normalized) at pixel X, Y
                    #
                    #     depth_array = sensor_data['CameraDepth'].data
                    #     value_at_pixel = depth_array[Y, X]
                    #

                    # Now we have to send the instructions to control the vehicle.
                    # If we are in synchronous mode the server will pause the
                    # simulation until we send this control.

                    if  args.autopilot:

                        client.send_control(
                            steer=0,
                            throttle=0.8,
                            brake=0.0,
                            hand_brake=False,
                            reverse=False)

                    else:

                        # Together with the measurements, the server has sent the
                        # control that the in-game autopilot would do this frame. We
                        # can enable autopilot by sending back this control to the
                        # server. We can modify it if wanted, here for instance we
                        # will add some noise to the steer.

                        control = measurements.player_measurements.autopilot_control
                        control.steer += random.uniform(-0.05, 0.05)
                        client.send_control(control)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
