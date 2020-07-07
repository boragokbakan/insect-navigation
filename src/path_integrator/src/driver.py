#!/usr/bin/env python
import rospy
import numpy as np
import tf

from path_integrator.msg import HomeVector

from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu

WHEEL_RADIUS = 0.033 
WHEEL_BASE = 0.287

MAXIMUM_ANGULAR_SPEED = 1.82
MAXIMUM_LINEAR_SPEED = 0.26

DISTANCE_CAP = 4 #Distance at which laser scan readings should be capped [m]
class Driver:
    '''A class that implements a driver node for mazes.'''
    def __init__(self): 
        self.pitch, self.yaw = 0.0, 0.0
        self.lastScan = LaserScan()

        self.searching = False
        self.homing = False

        rospy.init_node("driver_node")
        rospy.on_shutdown(self.halt)

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.sub = rospy.Subscriber("/base_scan", LaserScan, self.laser_callback, queue_size=1)
        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        
        self.rate = rospy.Rate(10)

    def laser_callback(self,data):
        if self.lastScan is not None:
            ranges = np.array(data.ranges)
            length = len(ranges)
            ranges[np.isinf(ranges)] = DISTANCE_CAP

            print("linspace: %s, %s, %s"%((data.angle_min,data.angle_max,length)))
            angles = np.linspace(data.angle_min,data.angle_max,length)
            rospy.logdebug("integrator_callback: not null")


            cos,sin = np.cos(angles),np.sin(angles)
            #print("cos: %s, sin: %s"%(cos,sin)) 
            
            normalized_ranges = np.divide(ranges,DISTANCE_CAP)
            x,y = np.dot(cos.T,ranges),np.dot(sin.T,ranges)
            
            print("x: %s, y: %s"%(x,y))
        
            msg = Twist()
            rotation = np.arctan2(y,x)
            msg.angular.z = np.maximum(-1,np.minimum(1,MAXIMUM_ANGULAR_SPEED*rotation))


            #Stop if the rotation
            if(self.pitch>np.pi/16):
                msg.angular.z = 0

            msg.linear.x = np.minimum(MAXIMUM_LINEAR_SPEED,5.2*normalized_ranges[length/2])

            self.pub.publish(msg)

            print("Rotational Speed:",msg.angular.z, " Linear Speed:", msg.linear.x)

        else:
            rospy.logdebug("laser_callback: skipped")


            self.rate.sleep()

    def halt(self):
        print("Killing node...")
        msg = Twist()
        msg.angular.z = 0
        msg.linear.x = 0
        self.pub.publish(msg)

    def imu_callback(self, data):
        (_, self.pitch, self.yaw) = tf.transformations.euler_from_quaternion([data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w])

if __name__ == '__main__':
    Driver = Driver()
    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()

