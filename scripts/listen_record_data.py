#!/usr/bin/env python3

import numpy as np
import time
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Joy
from nav_msgs.msg import Odometry
import rospy
import cv2
import matplotlib.pyplot as plt
import message_filters
import os
import pickle
from termcolor import cprint
import subprocess
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from parse_utils import BEVLidar
import yaml

class ListenRecordData:
    def __init__(self, rosbag_play_process, config_path):
        self.rosbag_play_process = rosbag_play_process
        self.data = []
        lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
        odom = message_filters.Subscriber('/aft_mapped_to_init', Odometry)
        # joystick = message_filters.Subscriber('/bluetooth_teleop/joy', Joy)
        joystick = message_filters.Subscriber('/joystick', Joy)

        ts = message_filters.ApproximateTimeSynchronizer([lidar, odom, joystick], 100, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)

        # read the config file
        self.config = yaml.load(open(config_path, 'r'))
        print('Config file loaded!')
        print(self.config)

        self.data = {'pose': [], 'bevlidarimg': [], 'joystick': []}

        self.bevlidar_handler = BEVLidar(x_range=(-self.config['LIDAR_RANGE_METERS'], self.config['LIDAR_RANGE_METERS']),
                                         y_range=(-self.config['LIDAR_RANGE_METERS'], self.config['LIDAR_RANGE_METERS']),
                                         z_range=(0, self.config['LIDAR_HEIGHT_METERS']), resolution=self.config['RESOLUTION'])

    def callback(self, lidar, odom, joystick):
        """[callback function for the approximate time synchronizer]

        Args:
            lidar ([type]): [lidar ROS message]
            odom ([type]): [odometry ROS message]
        """

        # get the bev lidar image
        lidar_points = pc2.read_points(lidar, skip_nans=True, field_names=("x", "y", "z"))
        bev_lidar_image = self.bevlidar_handler.get_bev_lidar_img(lidar_points)

        # show the image
        # cv2.imshow('bev_lidar', bev_lidar_image)
        # cv2.waitKey(1)

        # get the localization data
        x = odom.pose.pose.position.z
        y = odom.pose.pose.position.x
        euler = R.from_quat([odom.pose.pose.orientation.x,
                             odom.pose.pose.orientation.y,
                             odom.pose.pose.orientation.z,
                             odom.pose.pose.orientation.w]).as_euler('xyz', degrees=True)
        # get the yaw value from euler rotation angles
        yaw = euler[1]

        # get joystick data
        joy_axes = joystick.axes
        joy_buttons = joystick.buttons

        # save the data
        self.data['pose'].append([x, y, yaw])
        self.data['bevlidarimg'].append(bev_lidar_image)
        self.data['joystick'].append([joy_axes, joy_buttons])

    def save_data(self, data_path):
        """ convert joystick values to cmd_vel """
        cprint('Converting joystick values to cmd_vel.. ', 'yellow')
        for i in tqdm(range(len(self.data['joystick']))):
            joy_axes = self.data['joystick'][i][0]
            joy_buttons = self.data['joystick'][i][1]

            linear_x = self.joystickValue(joy_axes[self.config['kXAxis']], -self.config['kMaxLinearSpeed'])
            linear_y = self.joystickValue(joy_axes[self.config['kYAxis']], -self.config['kMaxLinearSpeed'])
            angular_z = self.joystickValue(joy_axes[self.config['kRAxis']], -np.deg2rad(90.0))

            cmd_vel = [linear_x, linear_y, angular_z]
            self.data['joystick'][i] = cmd_vel
        cprint('Joystick values converted to cmd_vel ', 'blue', attrs=['bold'])


        print('Number of data points : ', len(self.data['pose']))
        print('Saving data to : ', data_path)
        pickle.dump(self.data, open(data_path, 'wb'))
        cprint('Done!', 'green')

    @staticmethod
    def joystickValue(x, scale):
        kDeadZone = 0.02
        if abs(x) < kDeadZone: return 0.0
        return ((x - np.sign(x) * kDeadZone) / (1.0 - kDeadZone) * scale)

if __name__ == '__main__':
    rospy.init_node('listen_record_data', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    robot_name = rospy.get_param('robot_name')
    print('rosbag_path: ', rosbag_path)
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError('ROS bag file not found')

    # find root of the ros node and config file path
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(package_root, 'config/'+str(robot_name)+'.yaml')

    # start a subprocess to run the rosbag
    rosbag_play_process = subprocess.Popen(['rosbag', 'play', rosbag_path, '-r', '1', '--clock'])

    save_data_path = rosbag_path.replace('.bag', '_data.pkl')

    datarecorder = ListenRecordData(rosbag_play_process = rosbag_play_process,
                                    config_path = config_file_path)

    while not rospy.is_shutdown():
        #check if the python process is still running
        if rosbag_play_process.poll() is not None:
            print('rosbag process has stopped')
            datarecorder.save_data(save_data_path)
            exit(0)

    rospy.spin()






