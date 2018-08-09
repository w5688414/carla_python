#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time
import h5py
import numpy as np
import os
import sys
import termios
from tqdm import trange

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_2
    from pygame.locals import K_3
    from pygame.locals import K_4
    from pygame.locals import K_5
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from debugsaveimg import mkdir,fileCounter,press_any_key_exit,record_train_data


WINDOW_WIDTH = 200
WINDOW_HEIGHT = 150
MINI_WINDOW_WIDTH = 200
MINI_WINDOW_HEIGHT = 88
LIDAR_WIDTH = 200
LIDAR_HEIGHT = 200
NUM = 200

CAM_WINDOW_WIDTH = 800
CAM_WINDOW_HEIGHT = 600

record_dir = '../../data'


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=False,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=1,
        QualityLevel='Low')
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(CAM_WINDOW_WIDTH, CAM_WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(2.0, 0.0, 1.4)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(2.0, 0.0, 1.4)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)
    lidar = sensor.Lidar('Lidar32')
    lidar.set_position(0, 0, 2.5)
    lidar.set_rotation(0, 0, 0)
    lidar.set(
        Channels=0,
        Range=50,
        PointsPerSecond=100000,
        RotationFrequency=10,
        UpperFovLimit=0,
        LowerFovLimit=0)
    settings.add_sensor(lidar)
    return settings

def turbulence_halfsin(i, t, height=0.1):
    w = (np.pi/t)
    return height*np.sin(w*i)

def turbulence_fullTriangle(i,t,height=0.1):
    t = float(t)
    if i <= t/4:
        return 4*height/t*i
    elif t/4 <= i <= 3*t/4:
        return 2*height - 4*height/t*i
    else:
        return -4*height + 4*height/t*i



class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._display = None
        self._enable_autopilot = False
        self._is_on_reverse = False
        self._display_map = args.map
        self._city_name = None
        self._map = None
        self._map_shape = None
        self._position = None
        self._agent_positions = None

        self.pathdir = None
        self.f = None
        self.rgb_file = None
        self.seg_file = None
        self.depth_file = None
        self.lidar_file = None
        self.targets_file = None
        self.index_file = 200
        self._command = 2
        self.number_of_episodes = 2
        self.frames_per_cut = 4200
        self._joystick_control = 0
        self._turbulence_start_frame = 0
        self._turbulence_stop_frame = 0
        self._turbulence = 0
        self._turbulence_sym = 0

    def execute(self):
        """Launch the PyGame"""
        pygame.init()
        if pygame.joystick.get_count() > 0:
            self._joystick_control = 1
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print("Please make sure the joystick is on during the whole game!")
        else:
            self._joystick_control = 0
            print("You'd better plug the joystick in to improve your driving experience!")
        for episode in range(self.number_of_episodes):
            self._creat_dir()
            self._initialize_game()

            for frame in trange(self.frames_per_cut):
                if self._check_pygame() == False:
                    return
                # add turbulence
                if frame % 300 == 0:
                    if np.random.rand() > 0.0:
                        self._turbulence_sym = random.randrange(-1, 2, 2)
                        self._turbulence_start_frame = frame
                        self._turbulence_stop_frame = frame + np.random.randint(20,40)
                    else:
                        self._turbulence_sym = 0
                        self._turbulence_start_frame = -1
                        self._turbulence_stop_frame = -1
                if frame == self._turbulence_start_frame:
                    print('turbulence start')
                if frame == self._turbulence_stop_frame:
                    print('turbulence stop')
                if self._turbulence_start_frame < frame < self._turbulence_stop_frame :
                    self._turbulence = self._turbulence_sym * turbulence_fullTriangle(frame - self._turbulence_start_frame,
                                                                                          self._turbulence_stop_frame - self._turbulence_start_frame,
                                                                                          height=0.15)
                else:
                    self._turbulence = 0
                # if frame == 60:
                    # print('Record process will start in 40 frames')

                if frame % 10 == 0 and frame > 100:
                    self._on_loop(record=True, turbulence=self._turbulence)
                else:
                    self._on_loop(record=False, turbulence=self._turbulence)

        pygame.quit()

    def _creat_dir(self):
        # record
        mkdir(record_dir)
        num = len([name for name in os.listdir(record_dir) if name.startswith('episode')])
        formattednum = 'episode_{:0>3}'.format(num)
        self.pathdir = os.path.join(record_dir, formattednum)
        mkdir(self.pathdir)

    def _create_file(self):
        if (self.index_file == 200):
            num = fileCounter(self.pathdir)
            filename = "data_{:0>6}.h5".format(num)
            filepath = self.pathdir + '/' + filename
            self.f = h5py.File(filepath, "w")
            self.rgb_file = self.f.create_dataset("rgb", (NUM, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, 3), np.uint8)
            self.seg_file = self.f.create_dataset("CameraSemSeg", (NUM, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH ), np.uint8)
            self.depth_file = self.f.create_dataset("CameraDepth", (NUM, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH ), np.uint8)
            self.lidar_file = self.f.create_dataset('Lidar32', (NUM, LIDAR_HEIGHT, LIDAR_WIDTH ), np.uint8)
            self.targets_file = self.f.create_dataset("targets", (NUM, 28), np.float32)
            self.index_file = 0

    def _check_pygame(self):
        for event in pygame.event.get():
            if self._joystick_control and event.type == pygame.JOYBUTTONUP and event.button == 0:
                self._is_on_reverse = not self._is_on_reverse
            if event.type == pygame.QUIT:
                return False
            if pygame.key.get_pressed()[K_ESCAPE]:
                return False
            # if self._joystick_control and pygame.joystick.Joystick(0).get_button(5) and pygame.joystick.Joystick(0).get_button(7):
            #     return False
            if self._joystick_control and event.type == pygame.JOYBUTTONUP and event.button == 1:
                return False
            return True

    def _initialize_game(self):
        self._on_new_episode()
        if self._joystick_control == 0:
            pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        logging.debug('pygame started')

    def _on_new_episode(self):
        # self._carla_settings.randomize_seeds()
        # self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        if self._display_map:
            self._city_name = scene.map_name
        number_of_player_starts = len(scene.player_start_spots)
        # player_start = np.random.randint(number_of_player_starts)
        player_start = 122
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._is_on_reverse = False

    def _on_loop(self,record = False, turbulence=0):
        measurements, sensor_data = self.client.read_data()

        if self._joystick_control == 1:
            # control = self._get_XBOX_control(pygame.joystick.Joystick(0))
            control = self._get_joystick_control(pygame.joystick.Joystick(0))
        else:
            control = self._get_keyboard_control(pygame.key.get_pressed())

        # this code means first record keyboard or joystick, then add turbulence to it, and send control to server.
        if record :
            self._on_record(measurements,sensor_data,control)

        if turbulence != 0:
            control.steer += turbulence

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _on_record(self, measurements, sensor_data, control):
        self._create_file()
        if sensor_data and 'CameraRGB' in sensor_data \
                and 'CameraDepth' in sensor_data \
                and 'CameraSemSeg' in sensor_data \
                and 'Lidar32' in sensor_data:
            sensors, targets_data = record_train_data(measurements, sensor_data)
            self.rgb_file[self.index_file, :, :, :] = sensors['rgb']
            self.seg_file[self.index_file, :, :] = sensors['CameraSemSeg']
            self.depth_file[self.index_file, :, :] = sensors['CameraDepth']
            self.lidar_file[self.index_file, :, :] = sensors['Lidar32']
            self.targets_file[self.index_file, :] = targets_data
            self.targets_file[self.index_file, 24] = self._command  # 24代表的是command
            if self._enable_autopilot:
                pass
            else:
                self.targets_file[self.index_file, 0] = control.steer
                self.targets_file[self.index_file, 1] = control.throttle
                self.targets_file[self.index_file, 2] = control.brake
                self.targets_file[self.index_file, 3] = control.hand_brake
                self.targets_file[self.index_file, 4] = self._is_on_reverse
            self.index_file += 1

    def _get_XBOX_control(self, joystick):
        control = VehicleControl()
        control.throttle = 0.5 * (joystick.get_axis(5) + 1)

        control.brake = 0.5 * (joystick.get_axis(2) + 1)

        control.steer = joystick.get_axis(0)
        control.steer = 0.8 * control.steer

        control.reverse = self._is_on_reverse
        return control

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -0.5
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 0.5
        if keys[K_UP] or keys[K_w]:
            control.throttle = 0.7
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        if keys[K_2]:
            self._command = 2
        if keys[K_3]:
            self._command = 3
        if keys[K_4]:
            self._command = 4
        if keys[K_5]:
            self._command = 5
        control.reverse = self._is_on_reverse
        return control

    def _get_joystick_control(self,joystick):
        control = VehicleControl()
        tmp1 = 0.6 * joystick.get_axis(1)



        if (tmp1 <= 0):
            control.throttle = -tmp1
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = tmp1

        control.steer = joystick.get_axis(2)
        control.steer = 0.5 * control.steer * control.steer * control.steer
        # print('steer....',control.steer)

        #provide a stable autopilot
        autopilot = joystick.get_button(0)
        if autopilot == 1:
            self._enable_autopilot = not self._enable_autopilot

        # provide a stable reverse
        reverse = joystick.get_button(2)
        if reverse == 1:
            self._is_on_reverse = not self._is_on_reverse

        control.reverse = self._is_on_reverse
        return control

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
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
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster')
    argparser.add_argument(
        '-m', '--map',
        action='store_true',
        help='plot the map of the current city')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # print(__doc__)
    # os.system("gnome-terminal -e 'bash -c \"cd /home/kadn/AUTODRIVING/carla_python && ./CarlaUE4.sh -windowed -ResX=650 -ResY=350 -carla-server; exec bash\"'")
    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
