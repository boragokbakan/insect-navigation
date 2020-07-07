#!/usr/bin/env python
import rospy
import numpy as np
import tf

from path_integrator.msg import HomeVector
from sensor_msgs.msg import Imu

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse

#Control variables
IMU = True 
PITCH = False #Should the pitch be compensated for?
NOISE = False
PUBLISH_DIFF_VECTOR = False #Should we publish the delta step vector, or the integrated veridical home vector?

#Gaussian Noise Control Parameters
NOISE_SD = np.radians(1) #1 degree
NOISE_MEAN = 0

MAXIMUM_ANGULAR_SPEED = 0.91
MAXIMUM_LINEAR_SPEED = 0.26
class PathIntegrator:
    '''A class implementing a Path Integrator.'''
    WHEEL_RADIUS = 0.033 # meters
    WHEEL_BASE = 0.287 # meters
    
    def __init__(self):
        self.integrator, self.d_integrator = HomeVector(), HomeVector()
        self.integrator.x, self.integrator.y, self.integrator.orientation = (0.0,0.0,0.0) #Path Integrator
        self.d_integrator.x, self.d_integrator.y, self.d_integrator.orientation = (0.0,0.0,0.0) #Records relative changes at any time step, integrating this should similarly give true position
        
        self.lastJointStates = None

        self.homing = False

        self.trajectory_length = 0.0 #Accumulates step sizes

        self.old_yaw = None
        self.yaw = None

        self.yaw_history = []

        self.pitch = 0.0

        rospy.init_node("path_integrator_node")
        rospy.on_shutdown(self.halt)

        self.pub = rospy.Publisher("/home_vector", HomeVector, queue_size=1)

        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        #rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1) #True orientation from the simulator.
        rospy.Subscriber("/joint_states", JointState, self.integrator_callback, queue_size=1)

        #Steering commands
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Service("~/update_home", UpdateHome, self.learn_target)
        rospy.Service("~/set_homing", HomingSet, self.set_homing)

        self.rate = rospy.Rate(200)

    def integrator_callback(self,data):
        '''ROS Subscriber callback function for the /joint_states topic.
        
        Integrates agent movement.'''

        rospy.logdebug("integrator_callback: new data received")
        rospy.logdebug("p0: %s, p1: %s "%(data.position[0],data.position[1]))

        if self.lastJointStates is not None and self.old_yaw is not None:
            rospy.logdebug("integrator_callback: not null")

            wheel_r = data.position[0] - self.lastJointStates.position[0] #wheel rotation relative to last recording [rad]
            wheel_l = data.position[1] - self.lastJointStates.position[1] #wheel rotation relative to last recording [rad]

            rospy.logdebug("d0: %s, d1: %s "%(wheel_r,wheel_l))

            #Step size
            delta_s = PathIntegrator.WHEEL_RADIUS * (wheel_r + wheel_l) / 2.0 #meters
            
            self.trajectory_length += delta_s #Record total length
            if IMU:
                d_theta = self.yaw-self.old_yaw
            else:
                d_theta = PathIntegrator.WHEEL_RADIUS * (wheel_r - wheel_l) / PathIntegrator.WHEEL_BASE #rad 

            pitch = self.pitch
            if NOISE:
                d_theta += np.random.normal(NOISE_MEAN,NOISE_SD)
                pitch += np.random.normal(NOISE_MEAN,NOISE_SD)
            
            #If terrain shape is accounted for, multiply dx and dy by the cosine of the pitch angle, in order to project them to a 2D plane.
            if PITCH:
                dx = delta_s * np.cos(self.integrator.orientation + (d_theta / 2.0))*np.cos(pitch)
                dy = delta_s * np.sin(self.integrator.orientation + (d_theta / 2.0))*np.cos(pitch)
            else:
                dx = delta_s * np.cos(self.integrator.orientation + (d_theta / 2.0))
                dy = delta_s * np.sin(self.integrator.orientation + (d_theta / 2.0))

            self.integrator.x += dx
            self.integrator.y +=dy
            self.d_integrator.x = dx
            self.d_integrator.y = dx

            #If inertial measurement is used for bearing (i.e. gyroscope)
            if IMU:
                self.integrator.orientation = self.yaw
                self.d_integrator.orientation = self.yaw-self.old_yaw
            else:
                self.integrator.orientation += d_theta
                self.d_integrator.orientation = d_theta

            #The orientation is normalised from -PI to +PI
            self.integrator.orientation+=np.pi
            self.integrator.orientation%= np.pi*2
            self.integrator.orientation-= np.pi

            self.yaw_history.append((d_theta,self.yaw-self.old_yaw))
            
            if PUBLISH_DIFF_VECTOR:
                self.pub.publish(self.d_integrator)
            else:
                self.pub.publish(self.integrator)

            print("x: %s, y: %s, theta: %s, trajectory length = %s"%(self.integrator.x,self.integrator.y,self.integrator.orientation, self.trajectory_length))
        else:
            rospy.logdebug("integrator_callback: skipped")

        if self.homing:
            self.home()

        self.old_yaw = self.yaw
        self.lastJointStates = data
        self.rate.sleep()

    def odom_callback(self,data):
        (_, self.pitch, self.yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    def imu_callback(self, data):
        (_, self.pitch, self.yaw) = tf.transformations.euler_from_quaternion([data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w])

    def learn_target(self,req):
        '''Resets the Path Integrator, making the agent's location the new origin.'''
        self.integrator.x, self.integrator.y, self.integrator.orientation = (0.0,0.0,0.0) #The PI is reset
        return True

    def set_homing(self,req):
        '''ROS Service handler, stops the agent if not homing.'''
        self.homing = req.set
        if not self.homing:
            self.halt()
        return HomingSetResponse(self.homing)

    def home(self):
        '''Returns the agent to nest, publishes to /cmd_vel topic.'''
        x,y = self.integrator.x,self.integrator.y
        r,alpha = np.sqrt(x**2+y**2), np.arctan2(y,x)

        r = np.clip(r,-1,1)

        alpha = (alpha-np.pi)-(self.yaw)
        alpha = ((alpha+np.pi)%(np.pi*2))-np.pi 
        
        msg = Twist()
        msg.angular.z = np.minimum((20*alpha)/np.pi,MAXIMUM_ANGULAR_SPEED)
        msg.linear.x = MAXIMUM_LINEAR_SPEED*r 
        self.pub_cmd.publish(msg)

    def halt(self):
        '''Publishes a stop signal to cmd_vel topic.'''
        msg = Twist()
        msg.angular.z = 0
        msg.linear.x = 0
        self.pub_cmd.publish(msg)
if __name__ == '__main__':
    PI = PathIntegrator()
    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()
    print("Killing node...")
    np.savez("orientations.npz", orientation = PI.yaw_history) #Saves the orientation history as a pickle.
