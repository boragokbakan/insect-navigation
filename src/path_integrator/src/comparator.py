#!/usr/bin/env python
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from path_integrator.msg import QuickVector,HomeVector

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry


SYNC_WORLD_ODOM = False #True if PI sources should be recorded together to have identical number of recordings for both sources. False if they should be recorded as they come in. This should also be set to False if visual homing is running without PI.
PUBLISH_DIFF_VECTOR = False #Should we publish the difference vector, or the integrated veridical home vector?
QUICK_VECTOR = False
class Comparator:
    '''A helper class which records both predicted and actual position of the agent over time.'''
    def __init__(self): 
        rospy.init_node("comparator_node")

        #Polar PI
        if QUICK_VECTOR:
            self.integrator = QuickVector()
            self.integrator.r, self.integrator.orientation = (0.0,0.0)
            rospy.Subscriber("/quick_vector", QuickVector, self.quick_vector_callback, queue_size=1)
      
        #Cartesian PI
        else:
            rospy.Subscriber("/home_vector", HomeVector, self.home_vector_callback, queue_size=1)
            self.integrator = HomeVector()
            self.integrator.x, self.integrator.y, self.integrator.orientation = (0.0,0.0,0.0)
            self.position = None

        rospy.Subscriber("/odom", Odometry, self.odometry_callback, queue_size=1)

        #Gazebo Home Vector
        self.old_gazebo_odometry = HomeVector()

        #Lists to append home vectors as they are received.
        self.historyPI = []
        self.historyWorld = []

        #Distances to the origin by source of odometry
        self.euclidian_hv = 0.0
        self.euclidian_odom = 0.0

        self.rate = rospy.Rate(100)

    def quick_vector_callback(self,data):
        '''ROS Subscriber callback function for the /quick_vector topic.
        
        Records predicted position'''

        rospy.logdebug("quick_vector_callback: new data received")
        rospy.logdebug("r: %s, theta: %s "%(data.r,data.orientation))

        self.integrator.r = data.r
        self.integrator.orientation = data.orientation
        self.historyPI.append((data.r,data.orientation))

        print("Home Vector Length : %s, Odometry Length: %s, Hv/Odom: %s"%(data.r, self.euclidian_odom, (self.euclidian_hv/self.euclidian_odom)))

        if PUBLISH_DIFF_VECTOR:
            tmp = self.d_vector
        else:
            tmp = self.temporaryVector

        #If two sources are synchronized...
        if SYNC_WORLD_ODOM:
            self.historyWorld.append((tmp.x,tmp.y,tmp.orientation))

        self.old_gazebo_odometry = self.temporaryVector
        self.rate.sleep()

    def home_vector_callback(self,data):
        '''ROS Subscriber callback function for the /home_vector topic.
        
        Records predicted position'''

        rospy.logdebug("home_vector_callback: new data received")
        rospy.logdebug("x: %s, y: %s, theta: %s  "%(data.x,data.y,data.orientation))

        self.integrator.x = data.x
        self.integrator.y = data.y
        self.integrator.orientation = data.orientation

        #Append current PI state to history.
        self.historyPI.append((data.x,data.y,data.orientation))

        self.euclidian_hv = np.sqrt(np.square(data.x) + np.square(data.y))
        print("Home Vector Length : %s, Odometry Length: %s, Hv/Odom: %s"%(self.euclidian_hv, self.euclidian_odom, (self.euclidian_hv/self.euclidian_odom)))

        if PUBLISH_DIFF_VECTOR:
            tmp = self.d_vector
        else:
            tmp = self.temporaryVector

        #If two sources are synchronized...
        if SYNC_WORLD_ODOM:
            self.historyWorld.append((tmp.x,tmp.y,tmp.orientation))

        self.old_gazebo_odometry = self.temporaryVector
        
        self.rate.sleep()

    def odometry_callback(self,data):
        '''ROS Subscriber callback function for the /odom topic.
        
        Records real position provided by Gazebo.'''
        print("Gazebo /odom received")

        self.position = data
        self.temporaryVector = HomeVector()
        self.temporaryVector.x, self.temporaryVector.y = data.pose.pose.position.x, data.pose.pose.position.y

        #We extract orientation...
        _,_,self.temporaryVector.orientation = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        
        self.euclidian_odom = np.sqrt(np.square(self.temporaryVector.x) + np.square(self.temporaryVector.y))

        #If the path_integrator_node publishes step (diff) vectors (for analysis purposes), step vectors for Gazebo odometry are also calculated.
        if PUBLISH_DIFF_VECTOR:
            self.d_vector = HomeVector()
            self.d_vector.x = self.temporaryVector.x-self.old_gazebo_odometry.x 
            self.d_vector.y = self.temporaryVector.y-self.old_gazebo_odometry.y
            self.d_vector.orientation = self.temporaryVector.orientation-self.old_gazebo_odometry.orientation

            tmp = self.d_vector
            print("Diff vector x: %s, y: %s, theta: %s"%(tmp.x,tmp.y,tmp.orientation))

        else:
            tmp = self.temporaryVector

        #If two sources are not tethered...
        if not SYNC_WORLD_ODOM:
            self.historyWorld.append((tmp.x,tmp.y,tmp.orientation))
            self.old_gazebo_odometry = self.temporaryVector

        self.rate.sleep()

if __name__ == '__main__':
    Comparator = Comparator()
    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()
    print("Killing node...")

    file = "odometry_comparison3.npz"
    print("Saving to file: ",file)
    np.savez(file,hv_pi=Comparator.historyPI,hv_world=Comparator.historyWorld)

