#!/usr/bin/env python
import rospy
import numpy as np
import tf
import matplotlib.pyplot as plt

from central_complex import polar_to_cartesian, cartesian_to_polar

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from path_integrator.msg import CPU4, TB1
from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse


from sensor_msgs.msg import JointState

#Plots and records I_CPU4, r_CPU4 and r_TB1 messages
class CentralComplexPlot:
    '''A helper class that plots Central Complex messages on a polar plot. '''

    alpha = np.linspace(0,np.pi*15/4,16) #alpha represents the 8 preferred directions of the 16 TL neurons dividing the azimuth

    #Neuron firing rate encoding
    @staticmethod
    def activation(arr,a=1,b=0):
        return 1/(1+np.exp(-(a*arr-b)))

    #class constructor
    def __init__(self):
        self.yaw, self.pitch, self.roll = (0.0,0.0,0.0) #imu readings

        self.r_tb1 = np.zeros(8) #As TB1 neurons inhibit each other, integrator_callback should be able to see the previous value of it.
        self.I_cpu4 = np.repeat(0.5,16)  #Accumulator CPU4 cells are started as 0.5 (neutral), so that they are in the middle of the sigmoid function.
        self.old_yaw = 0

        self.last_imu = None
        self.lastJointStates = None

        #Record cell activity history.
        self.I_cpu4_history = []
        self.r_cpu4_history = []
        self.r_tb1_history  = []


        rospy.init_node("cx_plot_node")

        self.Subscriber = rospy.Subscriber("/r_tb1", TB1, self.r_tb1_callback, queue_size=1)   
        self.Subscriber = rospy.Subscriber("/i_cpu4", CPU4, self.I_cpu4_callback, queue_size=1)
        self.Subscriber = rospy.Subscriber("/r_cpu4", CPU4, self.r_cpu4_callback, queue_size=1)        

        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        rospy.Service("/plot_rcpu4", UpdateHome, self.plot_rcpu4)
        rospy.Service("/plot_icpu4", UpdateHome, self.plot_icpu4)
        rospy.Service("/plot_rtb1", UpdateHome, self.plot_rtb1)
        rospy.Service("/plot_all", UpdateHome, self.plot_all)        

        self.I_cpu4 = CPU4().CPU4
        self.r_cpu4 = CPU4().CPU4

        self.rate = rospy.Rate(10)

    def r_tb1_callback(self,data):
        self.r_tb1 = data.TB1
        self.r_tb1_history.append(data.TB1)
        #print("r_tb1 :", self.r_tb1)

    def I_cpu4_callback(self,data):
        self.I_cpu4 = data.CPU4
        self.I_cpu4_history.append(data.CPU4)

        if np.min(self.I_cpu4)<0.05 or np.max(self.I_cpu4)>0.95:
            rospy.loginfo("CPU4 is SATURATING!")

    def r_cpu4_callback(self,data):
        '''ROS Subscriber callback function, records the CPU4 layer's output.'''

        self.r_cpu4 = data.CPU4
        self.r_cpu4_history.append(data.CPU4)
        print("r_cpu4 :", self.r_cpu4)

    def imu_callback(self, msg):
        '''ROS Subscriber callback function, records the agent's yaw'''

        # Convert quaternions to Euler angles.
        (self.roll, self.pitch, self.yaw) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        #print("Roll: %s, pitch: %s, yaw: %s"%(self.roll,self.pitch,self.yaw))
        self.last_imu = msg

    def plot_rtb1(self,req):
        '''ROS Service server function, plots the compass layer.'''
        ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.plot(self.alpha[:8],self.r_tb1)
        ax.grid(True)
        ax.set_title("Output of TB1 Cells w/Different Preferred Angles")
        plt.show()
        return True

    def plot_icpu4(self,req):
        '''ROS Service server function, plots the input to CPU4 layer.'''
        ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.plot(self.alpha,self.I_cpu4)
        ax.grid(True)
        ax.set_title("Activation of CPU4 Cells w/Different Preferred Angles")
        plt.show()
        return True

    def plot_rcpu4(self,req):
        '''ROS Service server function, plots the output of CPU4 layer.'''
        ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.plot(self.alpha,self.r_cpu4)
        ax.grid(True)
        ax.set_title("Output of CPU4 Cells w/Different Preferred Angles")
        plt.show()
        return True

    def plot_all(self,req):
        '''ROS Service server function, plots everything.'''
        ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.plot(self.alpha,np.tile(self.r_tb1,2), label='r_TB1')
        ax.plot(self.alpha,self.I_cpu4,label='I_CPU4')
        ax.plot(self.alpha,self.r_cpu4, label='r_CPU4')

        x,y = polar_to_cartesian((self.I_cpu4,self.alpha))
        r,t = cartesian_to_polar([np.sum(x)/16,np.sum(y)/16])
        plt.scatter(t,r,marker='x',c='red')
    
        alpha = self.calculate_angle()
        ax.legend()
        ax.grid(True)
        
        ax.set_title("Different Cell Responses %s"%alpha)
        plt.show()    

        return True
    
    def calculate_angle(self):
        '''Helper function that computes the homing angle by shifting the CPU4 layer and aligning it to the compass layer. Not currently used.'''
        correlation = np.zeros(8)

        for i in range(8):
            correlation[i]+= np.dot(np.array(self.r_cpu4[i:8+i]).T,np.array(self.r_tb1))

        return np.argmax(correlation)

if __name__ == '__main__':
    CCP=CentralComplexPlot()
    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()
    print("Killing node...")
    np.savez("CentralComplexPlot.npz", r_tb1=CCP.r_tb1_history,I_cpu4=CCP.I_cpu4_history, r_cpu4=CCP.r_cpu4_history)
