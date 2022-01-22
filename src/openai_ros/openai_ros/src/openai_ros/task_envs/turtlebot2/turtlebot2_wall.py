import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from numpy import random
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
import wandb
from wandb.keras import WandbCallback
from .gazebo_connection import GazeboConnection
# Initialize your W&B project allowing it to sync with TensorBoard
wandb.init(name='PPO1', project="PPO Algorithm Testing", sync_tensorboard=True)

timestep_limit_per_episode = 1000 # Can be any Value
max_episode_step = 25000

register(
        id='MyTurtleBot2Wall-v0',
        entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_wall:TurtleBot2WallEnv',
        timestep_limit=timestep_limit_per_episode,
        max_episode_steps=max_episode_step, 
    )

class TurtleBot2WallEnv(turtlebot2_env.TurtleBot2Env):
    def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """
        self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim)
        
        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        # self.action_space = spaces.Box(numpy.array([0, -1]), numpy.array([1, 1]), dtype='float32')  # linear and angular speed
        # self.action_space = spaces.Box(-1., 1., shape=(1,), dtype='float32')  # linear and angular speed
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        
        
        self.number_observations = rospy.get_param('/turtlebot2/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')
        
        self.new_ranges = rospy.get_param('/turtlebot2/new_ranges')
        self.min_range = rospy.get_param('/turtlebot2/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot2/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot2/min_laser_value')
        
        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x
        self.desired_point.y
        self.desired_point.z
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        #num_laser_readings = len(laser_scan.ranges)/self.new_ranges
        # high = numpy.full((num_laser_readings), self.max_laser_value)
        # low = numpy.full((num_laser_readings), self.min_laser_value)
        high = numpy.full((self.number_observations), self.max_laser_value)
        low = numpy.full((self.number_observations), self.min_laser_value)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot2/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot2/turn_reward")
        self.distance_reward = rospy.get_param("/turtlebot2/distance_reward")
        self.end_episode_points = rospy.get_param("/turtlebot2/end_episode_points")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot2WallEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """s
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        self.total_path_lenght = 0.0
        
        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)
        
        # Get Desired Point to Get
        #self.allowed_x = list(range(-10, 96))   # testing_env_2_2
        self.allowed_x = list(range(-10, 87))   # irl_test4
        #self.allowed_x = list(range(-40, 54))   # lab_space
        # self.allowed_x = list(range(-85, 85))   # irl_test7
        #self.allowed_x = list(range(-45, 45))   # small_office
        #self.allowed_x = list(range(-40, 40))   # empty_coffe
        #self.allowed_x = list(range(-20, 25))   # test8
        self.x_random = (random.choice(self.allowed_x))/10.0
        #self.x_choose = self.x_random            # lab_space

        #self.allowed_y = list(range(-17, 98))  # testing_env_2_2
        self.allowed_y = list(range(-55, 100))  # irl_test4
        #self.allowed_y = list(range(-34, 10))   # lab_space
        # self.allowed_y = list(range(-85, 85))   # irl_test7
        #self.allowed_y = list(range(-125, 15))   # irl_test7
        #self.allowed_y = list(range(-105, 60))  # empty_coffe
        #self.allowed_y = list(range(-20, 22))  # test8
        self.y_random = (random.choice(self.allowed_y))/10.0
        #self.y_choose = self.y_random           # lab_space

        #irl_test3
        
        # if -0.5 < self.x_random < 0.5 and -0.5 < self.y_random < 0.5:
        #     self.x_choose = 1.0
        #     self.y_choose = 1.0
        # elif 1.0 < self.x_random < 4.0 and 5.0 < self.y_random < 8.0:
        #     self.x_choose = 3.5
        #     self.y_choose = 5.5
        # elif 0.0 < self.x_random < 2.0 and 2.5 < self.y_random < 3.5:
        #     self.x_choose = 3.0
        #     self.y_choose = 3.0
        # elif -1.8 < self.x_random < 3.8 and 3.5 < self.y_random < 3.8:
        #     self.x_choose = 1.5
        #     self.y_choose = 7.5
        # elif 3.5 < self.x_random < 3.75 and 0.2 < self.y_random < 3.8:
        #     self.x_choose = 1.5
        #     self.y_choose = 5.5
        # else:
        #     self.x_choose = self.x_random
        #     self.y_choose = self.y_random

                # irl_test4
        if -0.5 < self.x_random < 0.5 and -0.5 < self.y_random < 0.5:
            self.x_choose = 1.0
            self.y_choose = 1.0
        elif 4.0 < self.x_random < 7.0 and -3.85 < self.y_random < -4.1:
            self.x_choose = 7.5
            self.y_choose = -1.5
        elif 6.9 < self.x_random < 7.0 and -4.2 < self.y_random < 0.7:
            self.x_choose = 7.5
            self.y_choose = -1.5
        elif 4.65 < self.x_random < 5.3 and -2.25 < self.y_random < -1.6:
            self.x_choose = 7.5
            self.y_choose = -1.5
        elif -1.0 < self.x_random < -0.1 and -3.3 < self.y_random < -2.7:
            self.x_choose = -0.6
            self.y_choose = -3.5
        elif 1.7 < self.x_random < 2.2 and 2.8 < self.y_random < 3.35:
            self.x_choose = 3.0
            self.y_choose = 3.0
        elif 5.9 < self.x_random < 8.8 and 3.5 < self.y_random < 3.8:
            self.x_choose = 8.5
            self.y_choose = 3.0
        elif 4.5 < self.x_random < 7.5 and 5.6 < self.y_random < 8.5:
            self.x_choose = 6.7
            self.y_choose = 6.3
        elif 1.0 < self.x_random < 3.2 and 5.1 < self.y_random < 7.2:
            self.x_choose = 7.5
            self.y_choose = -1.5
        else:
            self.x_choose = self.x_random
            self.y_choose = self.y_random



                # irl_test7
        # if 5.7 < self.x_random < 6.0 and -8.0 < self.y_random < 0.4:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif -7.2 < self.x_random < 1.2 and -8.1 < self.y_random < -7.8:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif -3.0 < self.x_random < -1.0 and 9.0 < self.y_random < 10.5:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif -6.0 < self.x_random < -4.5 and 7.5 < self.y_random < 9.0:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif -9.4 < self.x_random < -8.2 and 7.3 < self.y_random < 8.5:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif 4.4 < self.x_random < 5.5 and 8.3 < self.y_random < 9.6:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif -3.5 < self.x_random < -2.2 and 3.5 < self.y_random < 4.8:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif 1.3 < self.x_random < 2.6 and 8.3 < self.y_random < 9.7:
        #     self.x_choose = -3.0
        #     self.y_choose = 9.0
        # elif 5.5 < self.x_random < 6.5 and 5.5 < self.y_random < 6.5:
        #     self.x_choose = -3.0
        #     self.y_choose = -9.0
        # elif 7.2 < self.x_random < 8.3 and 2.3 < self.y_random < 3.7:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif -8.7 < self.x_random < -7.4 and 0.4 < self.y_random < 1.7:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif -3.3 < self.x_random < -2.0 and -0.7 < self.y_random < 0.6:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif -10.0 < self.x_random < -8.7 and -3.1 < self.y_random < -1.8:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif -5.5 < self.x_random < -4.1 and -3.5 < self.y_random < -2.3:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif -6.6 < self.x_random < -5.5 and -6.3 < self.y_random < -5.2:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif 3.0 < self.x_random < 4.0 and -7.2 < self.y_random < -6.0:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # elif 8.0 < self.x_random < 9.0 and -4.5 < self.y_random < -3.2:
        #     self.x_choose = 7.0
        #     self.y_choose = -4.0
        # else:
        #     self.x_choose = self.x_random
        #     self.y_choose = self.y_random


        self.desired_point = Point()
        self.desired_point.x = float(self.x_choose)
        self.desired_point.y = float(self.y_choose)

        # goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
        # goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]
        # self.index = random.randrange(0, 13)
        # self.index = self.index + 1
        # self.desired_point.x = goal_x_list[self.index]
        # self.desired_point.y = goal_y_list[self.index]

        # specific point

        # self.desired_point.x = 6.0    # pedestrian env
        # self.desired_point.y = 0.0

        # self.desired_point.x = 7.0      # testing_env_2_2
        # self.desired_point.y = 9.0


        # self.desired_point.x = 0.0      # testing_env_2_2
        # self.desired_point.y = 2.0

        # self.desired_point.x = 4.0      # testing_env_2_2
        # self.desired_point.y = -1.0

        self.desired_point.z = 0.0
        self.show_marker_in_rviz(self.desired_point)

        # Get previous rotation to goal
        odometry_array = self.odom_sumary()
        current_position = Point()
        current_position.x = odometry_array[0]
        current_position.y = odometry_array[1]
        current_position.z = 0.0
        orientation_robot_to_goal = math.atan2(self.desired_point.y - current_position.y,self.desired_point.x - current_position.x)
        self.previous_rotation_to_goal_diff = self.difference_two_angles(orientation_robot_to_goal,odometry_array[2])
        self.prev_x = current_position.x
        self.prev_y = current_position.y



    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv

        # if action == 0: #FORWARD
        #     linear_speed = self.linear_forward_speed
        #     angular_speed = 0.0
        #     self.last_action = "FORWARDS"
        # elif action == 1: #LEFT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = self.angular_speed
        #     self.last_action = "TURN_LEFT"
        # elif action == 2: #RIGHT
        #     linear_speed = self.linear_turn_speed
        #     angular_speed = -1*self.angular_speed
        #     self.last_action = "TURN_RIGHT"

        if action == 0: #FORWARD
            linear_speed = 0.7
            angular_speed = 1.25
            #self.last_action = "FORWARDS"
        elif action == 1:
            linear_speed = 0.8
            angular_speed = 1
            #self.last_action = "TURN_LEFT"
        elif action == 2:
            linear_speed = 0.9
            angular_speed = 0.5
            #self.last_action = "TURN_RIGHT"
        elif action == 3:
            linear_speed = 1.0
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 4:
            linear_speed = 0.9
            angular_speed = -0.5
        elif action == 5:
            linear_speed = 0.8
            angular_speed = -1.0           
        elif action == 6:
            linear_speed = 0.7
            angular_speed = -1.25 

        # Continuous Action

        # linear_speed = 0.7
        # angular_speed = action[0]

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==>"+str(action))
    def odom_sumary(self):

        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        roll = pitch = robot_orientation = 0.0

        x_orientation = odometry.pose.pose.orientation.x
        y_orientation = odometry.pose.pose.orientation.y
        z_orientation = odometry.pose.pose.orientation.z
        w_orientation = odometry.pose.pose.orientation.w    #w: robot yaw
        orientation_list = [x_orientation, y_orientation, z_orientation, w_orientation]
        (roll, pitch, robot_orientation) = euler_from_quaternion(orientation_list)
        odometry_array = [x_position, y_position, robot_orientation]
        return odometry_array

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observationss
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")

        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        discretized_laser_scan = self.discretize_observation( laser_scan, self.new_ranges)

        obstacle_min_range = round(min(discretized_laser_scan), 2)
        #discretized_laser_scan_norm = scaler.fit_transform(discretized_laser_scan)

        # We round to only two decimals to avoid very big Observation space
        odometry_array = self.odom_sumary()
        current_position = Point()
        current_position.x = odometry_array[0]
        current_position.y = odometry_array[1]
        current_position.z = 0.0

        # Relationship between current_point with Goal_point
        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        #distance_from_des_point_norm = 1/self.get_distance_from_desired_point(current_position)
        orientation_robot_to_goal = math.atan2(self.desired_point.y - current_position.y,self.desired_point.x - current_position.x)
        rotation_to_goal_diff = self.difference_two_angles(orientation_robot_to_goal,odometry_array[2])
        #wandb.log({"Different of Rotation and Goal": rotation_to_goal_diff})


        # relationship_array = [distance_from_des_point,rotation_to_goal_diff]

        # relationship_array = [distance_from_des_point,rotation_to_goal_diff,obstacle_min_range]
        # relationship_array = [round(distance_from_des_point,2),round(rotation_to_goal_diff,2)]
        relationship_array = [round(distance_from_des_point,2),round(rotation_to_goal_diff,2), obstacle_min_range]


        # We only want the X and Y position and the Yaw

        #observations = discretized_laser_scan + relationship_array + odometry_array
        observations = numpy.concatenate((discretized_laser_scan, relationship_array), axis=0).tolist()


        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("TurtleBot2 is Too Close to wall==>")
        else:
            rospy.logerr("TurtleBot2 didnt crash at least ==>")
       
        # We round to only two decimals to avoid very big Observation space
            odometry_array = self.odom_sumary()

            current_position = Point()
            current_position.x = odometry_array[0]
            current_position.y = odometry_array[1]
            current_position.z = 0.0
            
            MAX_X = 25.0
            MIN_X = -25.0
            MAX_Y = 25.0
            MIN_Y = -25.0
            
            # We see if we are outside the Learning Space
            
            if current_position.x <= MAX_X and current_position.x > MIN_X:
                if current_position.y <= MAX_Y and current_position.y > MIN_Y:
                    rospy.logdebug("TurtleBot Position is OK ==>["+str(current_position.x)+","+str(current_position.y)+"]")
                    
                    # We see if it got to the desired point
                    if self.is_in_desired_position(current_position):
                        self._episode_done = True
                        #self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
                        self.gazebo.resetSim()
                else:
                    rospy.logerr("TurtleBot to Far in Y Pos ==>"+str(current_position.x))
                    self._episode_done = True
                    self.gazebo.resetSim()
            else:
                rospy.logerr("TurtleBot to Far in X Pos ==>"+str(current_position.x))
                self._episode_done = True
                self.gazebo.resetSim()
        return self._episode_done

    def _compute_reward(self, observations, done):
        # We round to only two decimals to avoid very big Observation space
        odometry_array = self.odom_sumary()

        current_position = Point()
        current_position.x = odometry_array[0]
        current_position.y = odometry_array[1]
        current_position.z = 0.0

        # Obstacles reward
        obstacle_min_range = observations[-1]

        # Distance Reward
        # distance_from_des_point_1 = self.get_distance_from_desired_point(current_position)
        # print('distance_from_des_point_1: ' + str(distance_from_des_point_1))
        distance_from_des_point = observations[-3]
        # print('distance_from_des_point: ' + str(distance_from_des_point))
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point
        # Orientation Reward
        orientation_robot_to_goal = math.atan2(self.desired_point.y - current_position.y, self.desired_point.x - current_position.x)
        # rotation_to_goal_diff_1 = self.difference_two_angles(orientation_robot_to_goal,odometry_array[2])
        # print('rotation_to_goal_diff_1: ' + str(rotation_to_goal_diff_1))
        rotation_to_goal_diff = observations[-2]
        # print('rotation_to_goal_diff: ' + str(rotation_to_goal_diff))
        rotations_cos_sum =  math.cos(rotation_to_goal_diff)     # [ -1 , 1]
        rotation_diff = math.fabs(math.fabs(self.previous_rotation_to_goal_diff) - math.fabs(rotation_to_goal_diff))   # [0 , pi]

        #r_orientation = (math.pi-math.fabs(rotation_to_goal_diff))/math.pi

        reward = -1.0

        path_lenght = 0.0
        distance_2_point = math.hypot(self.prev_x - current_position.x, self.prev_y - current_position.y) 
        start_to_goal_distance = math.hypot(self.desired_point.x, self.desired_point.y)

        if not done:


            # if observations[0] < 5*self.min_range or observations[1] < 5*self.min_range or observations[9] < 5*self.min_range or observations[10] < 5*self.min_range or observations[11] < 5*self.min_range or observations[19] < 5*self.min_range or observations[20] < 5*self.min_range:
            #     reward += -5.0
            if obstacle_min_range < 5*self.min_range:
                reward += -5.0

            #If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                distance_difference *= 4.0
            else:
                rospy.logerr("INCREASE IN DISTANCE BAD")
                distance_difference *= 2.0

            if math.fabs(rotation_to_goal_diff) > math.fabs(self.previous_rotation_to_goal_diff):
                rotation_diff *= -3.0
            else:
                rotation_diff *=  2.0

            reward += -1*distance_difference + (math.pi - math.fabs(rotation_to_goal_diff)) / math.pi + 3*rotations_cos_sum
            # reward += -1*distance_difference + rotation_diff
            # reward += -1*distance_difference + rotation_diff + 3*rotations_cos_sum
            # reward += -1*distance_difference + rotation_diff + rotations_cos_sum + ob_reward
            # reward += -1*distance_difference + (math.pi - math.fabs(rotation_to_goal_diff)) / math.pi + rotation_diff + 3*rotations_cos_sum
            # reward += rotation_diff # [ -3xpi , pi]
            # reward += (math.pi - math.fabs(rotation_to_goal_diff)) / math.pi
            # reward += (3*rotations_cos_sum) #[-3 , 3 ]
            distance_2_point = math.hypot(self.prev_x - current_position.x, self.prev_y - current_position.y) 
            path_lenght += distance_2_point
              
        else:
            
            if self.is_in_desired_position(current_position):
                reward = self.end_episode_points
            else:
                reward = -1.5*self.end_episode_points


        self.previous_distance_from_des_point = distance_from_des_point
        self.previous_rotation_to_goal_diff = rotation_to_goal_diff
        self.prev_x = current_position.x
        self.prev_y = current_position.y
        self.total_path_lenght += path_lenght
        #print('total_path_lenght: ' + str(self.total_path_lenght))



        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward

    def difference_two_angles(self,angle1, angle2):
        diff = (angle1 - angle2) % (math.pi * 2)
        if diff >= math.pi:
            diff -= math.pi * 2
        # diff = (diff - (-math.pi))/(math.pi - (-math.pi))
        return diff

    # Internal TaskEnv Methods
    
    def discretize_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        range_min = 0.10000000149
        range_max = 60.0

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("data=" + str(data))
        rospy.logwarn("new_ranges=" + str(new_ranges))
        rospy.logwarn("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                    # max_laser_value_norm = (self.max_laser_value - range_min)/(range_max - range_min)
                    # discretized_ranges.append(max_laser_value_norm)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                    # min_laser_value_norm = (self.min_laser_value - range_min)/(range_max - range_min)s
                    # discretized_ranges.append(min_laser_value_norm)
                else:
                    discretized_ranges.append(int(item))
                    # discretized_ranges.append(round(item,1))
                    # item_norm = (item - range_min)/(range_max - range_min)
                    # discretized_ranges.append(item_norm)
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                    #self.move_base(0.0, 0.0, epsilon=0.05, update_rate=10)
                    self.gazebo.resetSim()
                else:
                    rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    

        return discretized_ranges
        
        
    def is_in_desired_position(self,current_position, epsilon=0.5):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        is_in_desired_pos = False
        
        
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        
        return is_in_desired_pos
        
        
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """

        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        # v = numpy.linalg.norm(a) + numpy.linalg.norm(b)

        # if v != 0.0:
        #     distance = numpy.linalg.norm(a - b)
        #     distance = 1 - (numpy.linalg.norm(a - b)/v)
        # else:
        #     distance = 0.0
    
        return distance
