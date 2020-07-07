#!/usr/bin/env python
import rospy
import numpy as np
import tf
import cv2

from cv_bridge import CvBridge, CvBridgeError

#PI Services
from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse

#Sensor messages
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image

#Other messages..
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion, quaternion_from_euler


USE_CAMERA = False #Set true for camera based ALV, false for LiDAR
EDGE_LANDMARKS = False #Set true for edge vectors, false for landmark centroid vectors
HORIZON_SLICES = 360 #Divides the horizon in 360 slices of 1 degree

#Turtlebot3 driving parameters
MAXIMUM_ANGULAR_SPEED = 1.82
MAXIMUM_LINEAR_SPEED = 0.26
ROTATION_VELOCITY_CLIP = 0.2

#Camera parameters
CAMERA_LAG_COMPENSATION = 5 #slows rotations to compansate camera lag
SCAN_DISTANCE = 5 #How far should we scan from the panorama edge?

class ALV:
    '''A class implementing the Average Landmark Model proposed by Lambrinos et. al, 2000.'''
    def __init__(self): 
        self.yaw = 0.0

        rospy.init_node("alv_node")
        rospy.on_shutdown(self.halt)
        self.snapshots=[]
        self.target = None
        self.current_alv = None
        self.homing = True
        
        #If node started by a launch file, overwrite the USE_CAMERA parameter
        if rospy.has_param('camera_homing'):
            USE_CAMERA=rospy.get_param('camera_homing')

        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        
        if USE_CAMERA:
            rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback, queue_size=1)
        else:
            rospy.Subscriber("/base_scan", LaserScan, self.laser_callback, queue_size=1)

        rospy.Service("/update_target", UpdateHome, self.learn_target)
        rospy.Service("/set_homing", HomingSet, self.set_homing)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        #self.scan_range = 360 
        self.angles, self.increment = np.linspace(0,np.pi*2,HORIZON_SLICES+1, retstep=True)
        self.cv2_bridge = CvBridge()

        self.halt()
        self.rate = rospy.Rate(10)

    def set_homing(self,req):
        '''ROS Service handler, stops the agent if not homing.'''
        self.homing = req.set

        if not self.homing:
            self.halt()
        return HomingSetResponse(self.homing)
        
    def learn_target(self,req):
        '''ROS Service handler, saves the current ALV as the target.'''

        self.target = self.current_alv
        print("Updating target")
        return True

    def camera_callback(self,data):
        '''ROS Subscriber callback function for camera topic. As this function takes the average of the scene's illuminosity to extract landmarks, the landmarks should be darker than the background.
        
        @param data: camera image message.'''
        rospy.logdebug("camera_callback: new data received")

        width,height = data.width/2,data.height/2
        
        r = width-SCAN_DISTANCE #scanning radius

        print(width,height,r)
        try:
            cv_image = self.cv2_bridge.imgmsg_to_cv2(data,desired_encoding="bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            
            threshold = np.mean(cv_image) #Mean intensity is the threshold
            _,cv_image = cv2.threshold(cv_image,threshold,255,cv2.THRESH_BINARY) #Anything lighter than the threshold is masked out
            #cv2.imshow("Image window", cv_image)
            dest_img = np.zeros(HORIZON_SLICES)

            for i in range(HORIZON_SLICES):
                x = int(np.cos(np.radians(i))*r)
                y = int(np.sin(np.radians(i))*r)
                print(x,y)
                dest_img[(HORIZON_SLICES-1)-i] = cv_image[width+x][height+y]

        except CvBridgeError as e:
            print(e)

        #rotate = int(self.scan_range*self.yaw/(np.pi*2))
        #dest_img = np.roll(dest_img,rotate)
        #a = dest_img.reshape((1,self.scan_range))
        #cv2.imshow("Image window", np.tile(a,(5,1)))
        #cv2.waitKey(13)
        #print("Shape is = ",a.shape)

        dest_img[dest_img==255] = float("inf")

        landmarks = self.segment(dest_img,edge_vectors=False)
        self.computeALV(landmarks)
        print(landmarks)

        if self.target is None:
            self.target=self.current_alv
        
        if self.homing:
            self.home()

        self.rate.sleep()

    def laser_callback(self,data):
        '''ROS Subscriber callback function for laser scan topic.
        
        @param data: the laser scan message.'''

        rospy.logdebug("laser_callback: new data received")

        print("")
        rotate = int(HORIZON_SLICES*self.yaw/(np.pi*2))
        current_snapshot = np.array(data.ranges)
        self.snapshots.append(np.roll(current_snapshot,rotate))
        landmarks = self.segment(data.ranges,edge_vectors=EDGE_LANDMARKS)

        print("Landmarks : ", landmarks)
        self.computeALV(landmarks)

        #If there is no target, we record the current location.
        if self.target is None:
            self.target=self.current_alv

        #If the "emotional" state of the agent is homing, a cmd_vel message is published.
        if self.homing:
            self.home()

        self.rate.sleep()

    def segment(self, horizon, edge_vectors=False):
        '''Segment the horizon into landmarks.
        
        @param horizon: the scan vector of the horizon.
        
        @param edge_vectors: Setting this True assigns vectors to landmark edges. If False, landmark centers are assigned vectors.'''
        landmarks = []
            
        left_edge = float("nan") #this is the position of

        #Below we divide the panoramic scan into segments of landmarks

        #If we take edges as landmarks
        if(edge_vectors):
            for i in range(HORIZON_SLICES):
                if np.isinf(horizon[i-1]) != np.isinf(horizon[i]): #if we encounter a rise to inf or a fall to a number
                    landmarks.append((i,i))
            return landmarks #so much simpler than below
                
                    
        #If we take the center of objects as landmarks
        for i in range(HORIZON_SLICES):
            if(np.isnan(left_edge)): #if we are not on a landmark
                if not np.isinf(horizon[i]): #if we encounter a landmark
                    left_edge = i
            #if we are on a landmark        
            else:
                if np.isinf(horizon[i]):    
                    #print("Segment [%s,%s]"%(left_edge,i-1))
                    landmarks.append((left_edge,i-1))
                    left_edge = float("nan")

        #Below we check whether there is a landmark warping over the the scan array
        if not np.isnan(left_edge):
            if not np.isinf(horizon[i]):
                _, right_edge = landmarks[0]
                landmarks[0] = (left_edge,right_edge)
            else:
                landmarks.append(left_edge,i-1)
        
        return landmarks

    def computeALV(self, landmarks):
        '''Averages landmark vectors.
        
        @param landmarks: A tuple list of landmark edge angles, i.e. (left_edge,right_edge).'''
        alv = np.zeros(2)

        for landmark in landmarks:
            left_edge, right_edge = landmark
            steps = right_edge-left_edge

            if steps<0:
                steps+=HORIZON_SLICES #if a landmark wraps over the right edge to the beginning of the array, we pretend the array extends further

            angle = self.angles[left_edge]+(self.increment*steps/2.0)
            #print("inc : ",self.increment, "left e: ", left_edge, "steps: ",steps )
            #vector = ((1,angle))
            alv+=[np.cos(angle),np.sin(angle)] #Unit vector in cartesian
            #print("appending: ",vector)

        alv/=len(landmarks)
        theta = np.arctan2(alv[1],alv[0])
        theta += self.yaw

        print("yaw :",self.yaw)
        r = np.sqrt(alv[1]**2+alv[0]**2)
        self.current_alv = np.array([r,theta])

        print("ALV = ", self.current_alv)

    def home(self):
        '''Returns the agent to the target (i.e. nest) location. Publishes to /cmd_vel topic.'''
        slow_down = np.pi #If the difference between the target and the current angles is smaller than this, the turning rate will be adjusted proportionally
        
        diff = cartesian(self.current_alv)-cartesian(self.target)

        linear_velocity,alpha = polar(diff)

        alpha = alpha - self.yaw
        alpha = ((alpha+np.pi)%(np.pi*2))-np.pi #Gives the shortest angle between two points on the unit circle, so we don't make a full turn to go from 180 degrees to 179
        
        rotational_velocity = alpha/slow_down
        rotational_velocity = np.clip(rotational_velocity, -ROTATION_VELOCITY_CLIP, ROTATION_VELOCITY_CLIP)

        if USE_CAMERA:
            rotational_velocity = rotational_velocity/CAMERA_LAG_COMPENSATION

        if np.abs(alpha)>np.pi/8:
            linear_velocity = 0
        
        linear_velocity = np.clip(linear_velocity, -1, 1)
        print("alpha : ",alpha)
        msg = Twist()
        msg.angular.z = rotational_velocity*MAXIMUM_ANGULAR_SPEED
        msg.linear.x = linear_velocity*MAXIMUM_LINEAR_SPEED*np.cos(alpha)
        
        self.pub.publish(msg)

    def imu_callback(self, msg):
        '''ROS Subscriber callback function, records the agent's yaw.'''
        # Convert quaternions to Euler angles.
        (_,_,self.yaw) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def halt(self):
        msg = Twist()
        msg.angular.z = 0
        msg.linear.x = 0
        self.pub.publish(msg)
        np.savez("retinal.npz",snapshots=self.snapshots)

def cartesian(vector):
    '''Converts polar vectors to cartesian.'''
    r,theta = vector
    return np.array([r*np.cos(theta),r*np.sin(theta)])

def polar(vector):
    '''Converts cartesian vectors to polar.'''
    x,y = vector
    return np.array([np.sqrt(x**2+y**2),np.arctan2(y,x)])

if __name__ == '__main__':
    AverageLandmarkVector = ALV()

    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()

        print("Killing node...")

