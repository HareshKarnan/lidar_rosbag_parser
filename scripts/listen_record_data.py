#!/usr/bin/env python

import numpy as np
import time
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Joy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
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
import rosbag
import tf2_ros


def get_affine_mat(x, y, theta):
    """
    Returns the affine transformation matrix for the given parameters.
    """
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


class ListenRecordData:
    def __init__(self, rosbag_play_process, config_path, viz_lidar, odom_msgs=None, time_stamps=None):
        self.rosbag_play_process = rosbag_play_process
        self.data = []
        self.recorded_odom_msgs = odom_msgs
        self.recorded_time_stamps = np.asarray(time_stamps)

        lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
        # odom = message_filters.Subscriber('/aft_mapped_to_init', Odometry)
        # joystick = message_filters.Subscriber('/bluetooth_teleop/joy', Joy)
        joystick = message_filters.Subscriber('/joystick', Joy)
        odom = message_filters.Subscriber('/odom', Odometry)
        ts = message_filters.ApproximateTimeSynchronizer(
            [lidar, joystick, odom], 100, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)

        # read the config file
        self.config = yaml.load(open(config_path, 'r'))
        print('Config file loaded!')
        print(self.config)

        # publish global goal to move_base
        self.pub_goal = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=1)

        # self.data = {'pose': [], 'bevlidarimg': [], 'joystick': [], 'move_base_path': [], 'human_expert_odom': [],
        #              'odom': [], 'move_base_cmd_vel': []}
        self.data = {'pose': [], 'joystick': [], 'move_base_path': [], 'human_expert_odom': [],
                     'odom': [], 'move_base_cmd_vel': []}
        self.lidar_imgs = {}

        if self.config['robot_name'] == "spot":
            cprint('Processing rosbag collected on the SPOT',
                   'green', attrs=['bold'])
            self.data['odom_history'] = []

            # setup subsciber for odom
            self.odom_msgs = np.zeros((100, 6), dtype=np.float32)
            self.odom_sub = rospy.Subscriber(
                '/odom', Odometry, self.odom_callback, queue_size=1)

        elif self.config['robot_name'] == "jackal":
            cprint('Processing rosbag collected on the JACKAL',
                   'green', attrs=['bold'])

        self.bevlidar_handler = BEVLidar(x_range=(-self.config['LIDAR_RANGE_METERS'], self.config['LIDAR_RANGE_METERS']),
                                         y_range=(-self.config['LIDAR_RANGE_METERS'],
                                                  self.config['LIDAR_RANGE_METERS']),
                                         z_range=(-self.config['LIDAR_HEIGHT_METERS'],
                                                  self.config['LIDAR_HEIGHT_METERS']),
                                         resolution=self.config['RESOLUTION'], threshold_z_range=False)

        self.distance_travelled = None
        self.viz_lidar = viz_lidar
        self.start_time = None
        self.n = 0

        # setup subscriber for global path
        self.path_sub = rospy.Subscriber(
            '/move_base/TrajectoryPlannerROS/global_plan', Path, self.path_callback)
        self.move_base_path = None

        # setup subscriber for cmd_vel
        self.last_cmd_vel_callback, self.last_cmd_vel_callback_path = None, None
        self.cmd_vel_sub = rospy.Subscriber(
            '/cmd_vel', Twist, self.cmd_vel_callback)

        # setup tf2 publisher
        self.tf2_pub = tf2_ros.TransformBroadcaster()

    def callback(self, lidar, joystick, odom):
        """[callback function for the approximate time synchronizer]

        Args:
            lidar ([type]): [lidar ROS message]
            odom ([type]): [odometry ROS message]
        """

        # get the time of the current message in seconds
        if self.start_time is None:
            self.start_time = lidar.header.stamp.to_sec()
        current_time = lidar.header.stamp.to_sec()

        self.n += 1

        # publish the goal
        # find the closest message index in the recorded odom messages
        closest_index = np.searchsorted(
            self.recorded_time_stamps, current_time)+1
        future_index = min(closest_index + 30,
                           len(self.recorded_odom_msgs) - 1)
        odom_k = self.recorded_odom_msgs[future_index]
        for future_index in range(closest_index, len(self.recorded_odom_msgs)):
            odom_k = self.recorded_odom_msgs[future_index]
            dist = np.linalg.norm(np.array([odom.pose.pose.position.x, odom.pose.pose.position.y]) -
                                  np.array([odom_k.pose.pose.position.x, odom_k.pose.pose.position.y]))
            if dist > 10.0:
                break

        if self.n % 3 == 0:
            goal = self.convert_odom_to_posestamped_goal(odom_k)
            self.pub_goal.publish(goal)

        # get the bev lidar image
        lidar_points = pc2.read_points(
            lidar, skip_nans=True, field_names=("x", "y", "z"))
        bev_lidar_image = self.bevlidar_handler.get_bev_lidar_img(lidar_points)
        bev_lidar_image = self.convert_float64img_to_uint8(bev_lidar_image)

        # get the pose
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        ori = odom.pose.pose.orientation
        quaternion = [ori.x, ori.y, ori.z, ori.w]

        if self.distance_travelled is not None:
            self.distance_travelled += np.linalg.norm(
                np.asarray([x, y]) - np.asarray([self.data['pose'][-1][0], self.data['pose'][-1][1]]))

        if self.distance_travelled is None:
            self.distance_travelled = 0.0

        # convert quaternion to yaw angle
        yaw = R.from_quat(quaternion).as_euler('xyz', degrees=True)[2]

        # get joystick data
        joy_axes = joystick.axes
        linear_x = self.joystickValue(
            joy_axes[self.config['kXAxis']], -self.config['kMaxLinearSpeed'])
        linear_y = self.joystickValue(
            joy_axes[self.config['kYAxis']], -self.config['kMaxLinearSpeed'])
        angular_z = self.joystickValue(
            joy_axes[self.config['kRAxis']], -np.deg2rad(90.0), kDeadZone=0.0)

        # append to the data
        self.data['pose'].append([x, y, yaw])
        self.data['joystick'].append([linear_x, linear_y, angular_z])

        # save BEVLidar images to disk instead of pkl file
        self.lidar_imgs[self.n] = bev_lidar_image

        # # if using spot, then also record the past 1 sec odom data in self.data
        if self.config['robot_name'] == "spot":
            self.data['odom_history'].append(self.odom_msgs.flatten())

        # save the move_base_path
        if self.move_base_path is not None and (self.move_base_path_time - current_time) < 0.5:
            move_base_path = self.move_base_path_to_list(self.move_base_path)
            self.data['move_base_path'].append(move_base_path)
        else:
            cprint("move base path not available", "red", attrs=["bold"])
            self.data['move_base_path'].append(None)

        # save the human expert path
        human_expert_path = self.odom_msg_list_to_list(
            self.recorded_odom_msgs[closest_index:future_index])
        self.data['human_expert_odom'].append(human_expert_path)
        self.data['odom'].append([odom.pose.pose.position.x, odom.pose.pose.position.y,
                                 [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                                  odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]])

        # save the cmd_vel msg
        if self.last_cmd_vel_callback_path is None:
            self.data['move_base_cmd_vel'].append([None, None, None])
        else:
            self.data['move_base_cmd_vel'].append(
                [self.last_cmd_vel_callback.linear.x,
                 self.last_cmd_vel_callback.linear.y,
                 self.last_cmd_vel_callback.angular.z])

        # display the stuff
        if self.viz_lidar == "true":
            bev_lidar_image = cv2.cvtColor(bev_lidar_image, cv2.COLOR_GRAY2BGR)
            T_odom_robot = get_affine_matrix_quat(
                self.data['odom'][-1][0], self.data['odom'][-1][1], self.data['odom'][-1][2])
            for goal in self.data['human_expert_odom'][-1][:200]:
                T_odom_goal = get_affine_matrix_quat(goal[0], goal[1], goal[2])
                T_robot_goal = np.matmul(
                    np.linalg.pinv(T_odom_robot), T_odom_goal)
                T_c_f = [T_robot_goal[0, 2], T_robot_goal[1, 2]]
                t_f_pixels = [int(T_c_f[0] / 0.05) + 200,
                              int(-T_c_f[1] / 0.05) + 200]
                bev_lidar_image = cv2.circle(
                    bev_lidar_image, (t_f_pixels[0], t_f_pixels[1]), 1, (0, 0, 255), -1)

            cv2.imshow('bev_lidar', bev_lidar_image)
            cv2.waitKey(1)

    def path_callback(self, msg):
        """
        Callback for the global path
        """
        if self.start_time is not None:
            self.move_base_path = msg
            self.last_cmd_vel_callback_path = self.last_cmd_vel_callback
            self.move_base_path_time = msg.header.stamp.to_sec() - self.start_time

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel_callback = msg

    @staticmethod
    def move_base_path_to_list(move_base_path):
        move_base_path_list = []
        for goal in move_base_path.poses:
            move_base_path_list.append([goal.pose.position.x, goal.pose.position.y, [goal.pose.orientation.x,
                                                                                     goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w]])
        return move_base_path_list

    @staticmethod
    def odom_msg_list_to_list(odoms):
        tmp = []
        for odom in odoms:
            tmp.append([odom.pose.pose.position.x, odom.pose.pose.position.y,
                        [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]])
        return tmp

    def odom_callback(self, odom):

        tf = TransformStamped()
        tf.header.stamp = rospy.Time.now()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = odom.pose.pose.position.x
        tf.transform.translation.y = odom.pose.pose.position.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = odom.pose.pose.orientation.x
        tf.transform.rotation.y = odom.pose.pose.orientation.y
        tf.transform.rotation.z = odom.pose.pose.orientation.z
        tf.transform.rotation.w = odom.pose.pose.orientation.w
        self.tf2_pub.sendTransform(tf)

        self.odom_msgs = np.roll(self.odom_msgs, -1, axis=0)
        tmp = odom.twist.twist
        self.odom_msgs[-1] = np.array([tmp.linear.x, tmp.linear.y, tmp.linear.z,
                                       tmp.angular.x, tmp.angular.y, tmp.angular.z])

    def save_data(self, rosbag_path, save_data_path):
        """
        save processed ros bag as a pkl file and directory of bev lidar images

        rosbag_path: path to the rosbag that was processed
        save_data_path: location to place pkl file and bev lidar images directory

        example:
            rosbag_path: /home/abhinavchadaga/BCSAN/train.bag
            save_data_path: /home/abhinavchadaga/BCSAN/data

            BCSAN:
                - data:
                    - train.pkl
                    - train_data
                        - 1.png
                        - 2.png
                        ...

        """
        print('Number of data points : ', len(self.data['pose']))
        cprint('Distance travelled : ' +
               str(self.distance_travelled), 'green', attrs=['bold'])

        # generate path to pkl file and lidar data from
        # the rosbag file path and the save_data_path
        pkl_path = os.path.join(save_data_path, rosbag_path.split(
        '/')[-1].replace('.bag', '_data.pkl'))
        lidar_path = os.path.join(save_data_path, rosbag_path.split(
        '/')[-1].replace('.bag', '_data'))

        print('Saving pkl to : ', pkl_path)
        pickle.dump(self.data, open(pkl_path, 'wb'))

        print('Saving lidar images to: ', lidar_path)

        # create lidar data directory if necessary
        if not os.path.exists(lidar_path):
            os.makedirs(lidar_path)

        # write out each bev lidar image
        for timestamp, lidar_img in self.lidar_imgs.items():
            file_path = os.path.join(
                lidar_path, '{}.png'.format(timestamp))
            if not cv2.imwrite(file_path, lidar_img):
                raise Exception('Could not write image')
            print("saved to ", file_path)

        cprint('Done!', 'green')

    @staticmethod
    def convert_odom_to_posestamped_goal(odom):
        goal = PoseStamped()
        goal.header.frame_id = 'odom'
        goal.header.stamp = rospy.Time.now()
        orie = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        orie = R.from_quat(orie).as_euler('xyz', degrees=True)[2]
        orie = R.from_euler('xyz', [0, 0, orie], degrees=True).as_quat()
        goal.pose.position = odom.pose.pose.position
        goal.pose.position.z = 0.0
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = orie
        return goal

    @staticmethod
    def joystickValue(x, scale, kDeadZone=0.02):
        if kDeadZone != 0.0 and abs(x) < kDeadZone:
            return 0.0
        return ((x - np.sign(x) * kDeadZone) / (1.0 - kDeadZone) * scale)

    @staticmethod
    def convert_float64img_to_uint8(image):
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def get_affine_mat(x, y, theta):
        """
                Returns the affine transformation matrix for the given parameters.
                """
        theta = np.deg2rad(theta)
        return np.array([[np.cos(theta), -np.sin(theta), x],
                         [np.sin(theta), np.cos(theta), y],
                         [0, 0, 1]])


if __name__ == '__main__':
    rospy.init_node('listen_record_data', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    robot_name = rospy.get_param('robot_name')
    save_data_path = rospy.get_param('save_data_path')
    viz_lidar = rospy.get_param('viz_lidar')

    # parse the rosbag file and extract the odometry data
    cprint('First reading all the odom messages and timestamps from the rosbag',
           'green', attrs=['bold'])
    rosbag = rosbag.Bag(rosbag_path)
    # read all the odometry messages
    odom_msgs, time_stamps = [], []
    for topic, msg, t in tqdm(rosbag.read_messages(topics=['/odom'])):
        odom_msgs.append(msg)
        if len(time_stamps) == 0:
            time_stamps.append(0.0)
            start_time = t.to_sec()
        else:
            time_stamps.append(t.to_sec())
    cprint('Done reading odom messages from the rosbag !!!',
           'green', attrs=['bold'])

    if not os.path.exists(rosbag_path):
        cprint('rosbag path : ' + str(rosbag_path), 'red', attrs=['bold'])
        raise FileNotFoundError('ROS bag file not found')

    if not os.path.exists(save_data_path):
        cprint('Creating directory : ' +
               save_data_path, 'blue', attrs=['bold'])
        os.makedirs(save_data_path)
    else:
        cprint('Directory already exists : ' +
               save_data_path, 'blue', attrs=['bold'])

    # find root of the ros node and config file path
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(
        package_root, 'config/'+str(robot_name)+'.yaml')

    # start a subprocess to run the rosbag
    rosbag_play_process = subprocess.Popen(
        ['rosbag', 'play', rosbag_path, '-r', '1.0', '--clock'])

    datarecorder = ListenRecordData(rosbag_play_process=rosbag_play_process,
                                    config_path=config_file_path,
                                    viz_lidar=viz_lidar,
                                    odom_msgs=odom_msgs,
                                    time_stamps=time_stamps)

    while not rospy.is_shutdown():
        # check if the python process is still running
        if rosbag_play_process.poll() is not None:
            print('rosbag process has stopped')
            datarecorder.save_data(rosbag_path, save_data_path)
            print('Data was saved in :: ', save_data_path)
            exit(0)

    rospy.spin()
