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

    R            : restart level

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

try:
    import pygame
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
    camera0.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
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


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._display_map = args.map
        self._city_name = None
        self._map = None
        self._map_shape = None
        self._map_view = None
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
        self.number_of_episodes = 5
        self.frames_per_cut = 10000

    # def execute(self):
    #     """Launch the PyGame."""
    #     pygame.init()
    #     self._initialize_game()
    #     try:
    #         while True:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     return
    #             self._on_loop()
    #             self._on_render()
    #     finally:
    #         pygame.quit()


    def execute(self):
        """Launch the PyGame"""
        pygame.init()
        for episode in range(self.number_of_episodes):
            self.pathdir = '/home/kadn/dataTrain/episode_{:0>3}'.format(episode)
            mkdir(self.pathdir)
            self._initialize_game()

            for frame in range(0, self.frames_per_cut):
                if(self.index_file == 200):
                    self._create_newfile()
                # Read the data produced by the server this frame.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                # self._on_render()
        pygame.quit()

    def _create_newfile(self):
        num = fileCounter(self.pathdir)
        filename = "data_{:0>6}.h5".format(num)
        filepath = self.pathdir + '/' + filename
        self.f = h5py.File(filepath, "w")
        self.rgb_file = self.f.create_dataset("CameraRGB", (MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, NUM), np.uint8)
        self.seg_file = self.f.create_dataset("CameraSemSeg", (MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, NUM), np.uint8)
        self.depth_file = self.f.create_dataset("CameraDepth", (MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH, NUM), np.uint8)
        self.lidar_file = self.f.create_dataset('Lidar32', (LIDAR_HEIGHT, LIDAR_WIDTH, NUM), np.uint8)
        self.targets_file = self.f.create_dataset("targets", (NUM, 28), np.float32)
        self.index_file = 0

    def _initialize_game(self):
        self._on_new_episode()

        if self._city_name is not None:
            self._map = CarlaMap(self._city_name, 0.1643, 50.0)
            self._map_shape = self._map.map_image.shape
            self._map_view = self._map.get_map(WINDOW_HEIGHT)

            extra_width = int((WINDOW_HEIGHT/float(self._map_shape[0]))*self._map_shape[1])
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + extra_width, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
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
        player_start = 0
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):

        measurements, sensor_data = self.client.read_data()

        control = self._get_keyboard_control(pygame.key.get_pressed())

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)
        # if 'CameraRGB' in sensor_data:
        #     print('CameraRGB')
        # if 'CameraDepth' in sensor_data:
        #     print('CameraDepth')
        # if 'CameraSemSeg' in sensor_data:
        #     print('CameraSemSeg')
        # if 'Lidar32' in sensor_data:
        #     print('Lidar32')

        if sensor_data and 'CameraRGB' in sensor_data \
                and 'CameraDepth' in sensor_data \
                and 'CameraSemSeg' in sensor_data \
                and 'Lidar32' in sensor_data :
            sensors, targets_data = record_train_data(measurements, sensor_data)
            self.rgb_file[:, :, self.index_file] = sensors['CameraRGB']
            self.seg_file[:, :, self.index_file] = sensors['CameraSemSeg']
            self.depth_file[:, :, self.index_file] = sensors['CameraDepth']
            self.lidar_file[:, :, self.index_file] = sensors['Lidar32']
            self.targets_file[self.index_file, :] = targets_data
            self.targets_file[self.index_file, 24] = self._command   #24代表的是command
            if self._enable_autopilot:
                pass
            else:
                self.targets_file[self.index_file, 0] = control.steer
                self.targets_file[self.index_file, 1] = control.throttle
                self.targets_file[self.index_file, 2] = control.brake
                self.targets_file[self.index_file, 3] = control.hand_brake
                self.targets_file[self.index_file, 4] = self._is_on_reverse
            self.index_file += 1
            # print(self.index_file)



    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -0.7
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 0.7
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

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()

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

    print(__doc__)

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
