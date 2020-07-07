#!/usr/bin/env python
import rospy
import numpy as np
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from path_integrator.msg import QuickVector


from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse

MAXIMUM_ANGULAR_SPEED = 0.91
MAXIMUM_LINEAR_SPEED = 0.26
class QuickandDirty:
    '''A class implementing a Path Integrator as proposed by Mueller and Wehner, 1988.'''
    WHEEL_RADIUS = 0.033 # meters
    WHEEL_BASE = 0.287 # meters
    #class constructor
    def __init__(self):
        self.integrator = QuickVector()
        self.integrator.r, self.integrator.orientation = (0.0,0.0)
        self.yaw, self.pitch, self.roll = (0.0,0.0,0.0)

        self.old_yaw = None
        self.lastJointStates = None

        self.homing = False

        rospy.init_node("quick_and_dirty")

        self.pub = rospy.Publisher("/quick_vector", QuickVector, queue_size=1)
        
        #Steering commands
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Service("~/update_home", UpdateHome, self.learn_target)
        rospy.Service("~/set_homing", HomingSet, self.set_homing)

        rospy.Subscriber("/joint_states", JointState, self.integrator_callback, queue_size=1)
        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        self.rate = rospy.Rate(10)

    def integrator_callback(self,data):
        '''ROS Subscriber callback function, integrates the agent movement.'''
        rospy.logdebug("integrator_callback: new data received")

        if self.lastJointStates is not None and self.old_yaw is not None:
            rospy.logdebug("integrator_callback: not null")

            wheel_r = data.position[0] - self.lastJointStates.position[0] #rad
            wheel_l = data.position[1] - self.lastJointStates.position[1] #rad

            rospy.logdebug("d0: %s, d1: %s "%(wheel_r,wheel_l))

            delta_s = QuickandDirty.WHEEL_RADIUS * (wheel_r + wheel_l) / 2.0 # meters
            
            #d_theta = QuickandDirty.WHEEL_RADIUS * (wheel_r - wheel_l) / QuickandDirty.WHEEL_BASE #rad 
            d_theta = self.integrator.orientation-self.yaw

            self.integrator.r += delta_s * np.cos(d_theta)*np.cos(self.pitch)
            self.integrator.orientation += delta_s * np.sin(d_theta) / self.integrator.r
            self.integrator.orientation %= np.pi*2 #The orientation is wrapped over 2*PI, no need to integrate the orientation beyond range.

            self.pub.publish(self.integrator)

            #print("x: %s, y: %s, theta: %s"%(self.integrator.x,self.integrator.y,self.integrator.orientation))
        else:
            rospy.logdebug("integrator_callback: skipped")

        if self.homing:
            self.home()

        self.old_yaw = self.yaw
        self.lastJointStates = data
        self.rate.sleep()

    def imu_callback(self, msg):
        '''ROS Subscriber callback function, records the agent's yaw.'''
        # Convert quaternions to Euler angles.
        (self.roll, self.pitch, self.yaw) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        print("Roll: %s, pitch: %s, yaw: %s"%(self.roll,self.pitch,self.yaw))

    def learn_target(self,req):
        '''Resets the Path Integrator, making the agent's location the new origin.'''
        self.integrator.r, self.integrator.orientation = (0.0,0.0) #The PI is reset
        return True

    def set_homing(self,req):
        '''ROS Service handler, stops the agent if not homing.'''
        self.homing = req.set
        if not self.homing:
            self.halt()
        return HomingSetResponse(self.homing)

    def home(self):
        '''Returns the agent to nest, publishes to /cmd_vel topic.'''
        r,alpha = self.integrator.r, self.integrator.orientation

        r = np.clip(r,-1,1)

        #Calculate the rotation signal for the agent
        alpha = (alpha-np.pi)-(self.yaw)
        alpha = ((alpha+np.pi)%(np.pi*2))-np.pi #Get the shortest angular distance
        
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
    QuickandDirty()
    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()
    print("Killing node...")
