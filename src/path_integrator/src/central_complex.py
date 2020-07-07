#!/usr/bin/env python
import rospy
import numpy as np
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from path_integrator.msg import CPU4
from path_integrator.msg import TB1

from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

MAXIMUM_ANGULAR_SPEED = 1.82/4
WALKING_SPEED = 0.26
WHEEL_RADIUS = 0.033 #rospy.get_param("path_integrator/WHEEL_RADIUS") # 0.033 meters by default

#Gaussian Noise Control Parameters
NOISE_SD = 0.10 #baseline CPU4 activity = 0.5
NOISE_MEAN = 0


#Weights adapted from Le Moel et. al (2019) with some adjustments
tl_a = 3
tl_b = 0
cl1_a = 4
cl1_b = -1.5
tb1_a = 6.8
tb1_b = 3
cpu4_a = 5
cpu4_b = 2.5
cpu1_a = 6
cpu1_b = 2

class CentralComplex:
    '''An implementation of the Central Complex model based on Le Moel et. al (2019), proposed by Stone et. al, 2017.'''
    alpha = np.linspace(0,np.pi*15/4,16) #alpha represents the 8 preferred directions of the 16 TL neurons dividing the azimuth
    d = 0.33 #TB1 to TB1 inhibition scaling factor
    acc = 0.0002 #We keep a slow accumulation coefficient in order to prevent early saturation
    decay = 0.01 #.1

    #Layer 3 TB1: Compass Constants
    W_cl1 = np.zeros((16,8)) #W_CL1_TB1 represents the matrix mapping CL Neuron pairs onto 8 TB1 Neurons
    W_cl1[:8] += np.identity(8)
    W_cl1[8:] += np.identity(8)

    W_tb1 = d*(np.cos(alpha[:8,None]-alpha[:8])-1) #W_TBi_TBj 


    #Neuron firing rate encoding, sigmoid activation function.
    @staticmethod
    def activation(arr,a=1,b=0):
        return 1/(1+np.exp(-(a*arr-b)))

    #class constructor
    def __init__(self):
        self.yaw = 0.0 #agent heading

        self.r_tb1 = np.zeros(8) #As TB1 neurons inhibit each other, integrator_callback should be able to see the previous value of it.
        self.I_cpu4 = np.repeat(0.5,16)  #Accumulator CPU4 cells are started as 0.5 (neutral), so that they are in the middle of the sigmoid function.
        self.r_cpu4 = None

        #CPU1 steering
        self.cpu1_theta = 0

        self.W_vm = np.repeat(0.5,16)

        self.homing = False #Agent's motivational state
        self.lastJointStates = None #For stride integration

        rospy.init_node("cx_node")
        rospy.on_shutdown(self.halt)

        #Publish integrator and compass states for plotting
        self.pub_rcpu4 = rospy.Publisher("/r_cpu4", CPU4, queue_size=1)
        self.pub_icpu4 = rospy.Publisher("/i_cpu4", CPU4, queue_size=1)
        self.pub_rtb1 = rospy.Publisher("/r_tb1", TB1, queue_size=1)

        #Steering commands
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Subscriber("/joint_states", JointState, self.integrator_callback, queue_size=1)
        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        rospy.Service("~/update_home", UpdateHome, self.learn_target)
        rospy.Service("~/set_homing", HomingSet, self.set_homing)

        self.rate = rospy.Rate(10)
        print("Node set up...")

    def integrator_callback(self,data):
        '''ROS Subscriber callback function for the /joint_states topic.
        
        Integrates agent movement based on the CX model.'''

        rospy.logdebug("integrator_callback: new data received")
        rospy.logdebug("p0: %s, p1: %s "%(data.position[0],data.position[1]))

        if self.lastJointStates is not None and self.last_imu is not None:
            rospy.logdebug("integrator_callback: not null")

            wheel_r = data.position[0] - self.lastJointStates.position[0] #wheel rotation (rad) since last reading
            wheel_l = data.position[1] - self.lastJointStates.position[1] #wheel rotation (rad) since last reading

            rospy.logdebug("d0: %s, d1: %s "%(wheel_r,wheel_l))

            #Step size, TN input
            delta_s = WHEEL_RADIUS * (wheel_r + wheel_l) / 2.0 # meters
            delta_s *= 55 #We multiply by a coefficient to bring the TN output close to 1.

            #Layer 1 TN Neurons: Speed or Distance Input 
            phi = np.cos(np.pi/4)
            I_tn = np.array([phi,phi])*delta_s
            r_tn = np.minimum(1,np.maximum(0,I_tn)) #clipped between [0,1]
            print("I_tn", np.min(I_tn), np.max(I_tn),np.mean(I_tn))

            #Layer 1 TL Neurons: Directional Input
            alpha = np.linspace(0,np.pi*15/4,16) #alpha represents the 8 preferred directions of the 16 TL neurons dividing the azimuth
            I_tl = np.cos(alpha-self.yaw)#TL Neurons receive the cosine of the difference between their preferred direction (alpha) and the agent's heading
            r_tl = self.activation(I_tl,a=tl_a,b=tl_b) + np.random.normal(NOISE_MEAN,NOISE_SD,16)
            print("I_tl", np.min(I_tl), np.max(I_tl),np.mean(I_tl))

            #Layer 2 CL1 Neurons
            I_cl1 = -r_tl
            r_cl1 = self.activation(I_cl1,a=cl1_a,b=cl1_b) + np.random.normal(NOISE_MEAN,NOISE_SD,16)
            print("I_cl1", np.min(I_cl1), np.max(I_cl1),np.mean(I_cl1))
            print("r_cl1", np.min(r_cl1), np.max(r_cl1),np.mean(r_cl1))
            
            #Layer 3 TB1 Neurons: Compass
            I_tb1 = np.dot(self.W_cl1.T,r_cl1) + np.dot(self.W_tb1.T,self.r_tb1)
            self.r_tb1 = self.activation(I_tb1, a=tb1_a, b=tb1_b) + np.random.normal(NOISE_MEAN,NOISE_SD,8)
            print("I_tb1", np.min(I_tb1), np.max(I_tb1),np.mean(I_tb1))
            print("r_tb1", np.min(self.r_tb1), np.max(self.r_tb1),np.mean(self.r_tb1))

            #Layer 4 CPU4 Neurons: Speed Accumulation
            self.I_cpu4[:8] += self.acc * r_tn[0]*(1-self.r_tb1-self.decay)
            self.I_cpu4[8:] += self.acc * r_tn[1]*(1-self.r_tb1-self.decay)
            self.I_cpu4 = np.minimum(1,np.maximum(0,self.I_cpu4))
            print("I_cpu4", np.min(self.I_cpu4), np.max(self.I_cpu4),np.mean(self.I_cpu4))
            self.r_cpu4 = self.activation(self.I_cpu4,a=cpu4_a, b=cpu4_b) + np.random.normal(NOISE_MEAN,NOISE_SD,16)

            r,theta = self.get_centroid(self.r_cpu4,self.alpha,homing=True)

            print("Home vector r:",r, " alpha: ", theta)
            print("CPU4 max angle:", np.degrees(alpha[np.argmax(self.r_cpu4)]))

            #We publish CPU4 and TB1 layers for analysis
            self.pub_rcpu4.publish(list(self.r_cpu4))
            self.pub_icpu4.publish(list(self.I_cpu4))
            self.pub_rtb1.publish(list(self.r_tb1))

        else:
            rospy.logdebug("integrator_callback: skipped")

        if self.homing:
                self.home()

        #We save the encoder states...
        self.lastJointStates = data
        self.rate.sleep()

    def learn_target(self,req):
        '''Resets the CPU4 layer, making the agent's location the new origin.'''
        self.I_cpu4 = np.repeat(0.5,16) #setting all to the baseline activity 0.5 will make the current position of the agent the new center
        return True

    def set_homing(self,req):
        '''ROS Service handler, stops the agent if not homing.'''
        self.homing = req.set
        if not self.homing:
            self.halt()
        return HomingSetResponse(self.homing)

    def home(self):
        '''Returns the agent to nest, publishes to /cmd_vel topic.'''
        _,alpha = self.get_centroid(self.r_cpu4,self.alpha,homing=True) #we get the home vector's angle

        msg = Twist()
        msg.angular.z = np.minimum((20*alpha)/np.pi,MAXIMUM_ANGULAR_SPEED)
        msg.linear.x = WALKING_SPEED #Constant Homing Speed
        self.pub_cmd.publish(msg)

    def home_CPU1(self):
        '''Returns the agent to nest, publishes to /cmd_vel topic. Based on the CPU1 layer of the CX model. Currently unstable.'''
        #Layer 5 CPU1 Neurons: Steering
        cpu4_left = self.r_cpu4[:8] #left set, should be identical to right set (except noise)
        cpu4_right = self.r_cpu4[8:] #right set, should be identical to left set (except noise)

        cpu4_left = self.r_tb1*np.roll(cpu4_left,-1) #left set offset to left
        cpu4_right = self.r_tb1*np.roll(cpu4_right,1) #right set offset to right

        I_cpu1 = np.concatenate((cpu4_left,cpu4_right)).reshape(16) 
        r_cpu1 = self.activation(I_cpu1,a=cpu1_a,b=cpu1_b)

        #Compare left and right sets to produce a steering signal
        self.cpu1_theta += 0.05*(np.sum(r_cpu1[:8])-np.sum(r_cpu1[8:]))
        self.cpu1_theta = np.clip(self.cpu1_theta,-1,1)
        alpha = self.cpu1_theta
        print("theta :", self.cpu1_theta, "steering signal:",alpha)

        msg = Twist()
        msg.angular.z = np.minimum(alpha/np.pi,MAXIMUM_ANGULAR_SPEED)
        msg.linear.x = WALKING_SPEED #Constant Homing Speed
        self.pub_cmd.publish(msg)

    def halt(self):
        '''Publishes a stop signal to cmd_vel topic.'''
        msg = Twist()
        msg.angular.z = 0
        msg.linear.x = 0
        self.pub_cmd.publish(msg)

    def get_centroid(self,ranges,angles,homing=True,cartesian=False):
        '''Get the centroid of the CPU4 ring (i.e. home vector).

        @param homing: True if home vector angle should be adjusted for the agent's heading, False for the absolute angle

        @param cartesian: True if the home vector should be returned in cartesian coordinates. False by default.'''
        vector = np.array([ranges,angles])
        num = len(ranges)
        x,y = polar_to_cartesian(vector)
        r,alpha = cartesian_to_polar([np.sum(x)/num,np.sum(y)/num])

        if homing:
            alpha = (alpha-np.pi)-(self.yaw)
            alpha = ((alpha+np.pi)%(np.pi*2))-np.pi #Gives the shortest angle between two points on the unit circle, so we don't make a full turn to go from 180 degrees to 179

        if cartesian:
            return cartesian((r,alpha))
        return np.array([r,alpha])

    def imu_callback(self, msg):
        '''ROS Subscriber callback function, records the agent's yaw.'''
        # Convert quaternions to Euler angles.
        (_, _, self.yaw) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        #print("Roll: %s, pitch: %s, yaw: %s"%(self.roll,self.pitch,self.yaw))
        self.last_imu = msg

def polar_to_cartesian(vector):
    '''Converts polar vectors to cartesian.'''
    r,theta = vector
    return np.array([r*np.cos(theta),r*np.sin(theta)])

def cartesian_to_polar(vector):
    '''Converts cartesian vectors to polar.'''
    x,y = vector
    return np.array([np.sqrt(x**2+y**2),np.arctan2(y,x)])
        
if __name__ == '__main__':
    CCX = CentralComplex()

    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()

    CCX.halt() #We stop the agent.
    print("Node killed.")