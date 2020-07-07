#!/usr/bin/env python
import rospy
import numpy as np
import tf

from path_integrator.srv import UpdateHome, UpdateHomeResponse
from path_integrator.srv import HomingSet, HomingSetResponse

from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Twist

from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry

MAXIMUM_ANGULAR_SPEED = 1.82/8
MAXIMUM_LINEAR_SPEED = 0.26
TANGENTIAL_VECTOR_WEIGHT = 3
RADIAL_VECTOR_WEIGHT = 1
HORIZON_SLICES = 360 #Divides the horizon in 360 slices

class Snapshot:
    '''A class implementing the Snapshot model.'''

    def __init__(self): 
        self.yaw = 0.0

        rospy.init_node("snapshot_node")
        rospy.on_shutdown(self.halt)


        self.angles, self.increment = np.linspace(0,np.pi*2,HORIZON_SLICES+1, retstep=True)

        self.target_snapshot = None
        self.target_landmark_vectors = None
        self.target_gap_vectors = None
        

        self.current_snapshot = np.zeros(HORIZON_SLICES)

        self.homing = True

        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)
        rospy.Subscriber("/base_scan", LaserScan, self.laser_callback, queue_size=1)
        rospy.Service("/update_target", UpdateHome, self.update_target)
        rospy.Service("/set_homing", HomingSet, self.set_homing)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)


        self.halt()
        self.rate = rospy.Rate(10)

    def set_homing(self,req):
        self.homing = req.set
        if not self.homing:
            self.halt()
        return HomingSetResponse(self.homing)

    def laser_callback(self,data):
        '''ROS Subscriber callback function for laser scan topic.
        
        @param data: the laser scan message.'''
        rospy.logdebug("integrator_callback: new data received")
        #rospy.logdebug("p0: %s, p1: %s "%(data.position[0],data.position[1]))
        print("")

        rotate = int(HORIZON_SLICES*self.yaw/(np.pi*2)) #Rotate the laser scan to a standard orientation

        print("roll by : ", rotate)
        self.current_snapshot = np.array(data.ranges)
        self.current_snapshot = np.roll(self.current_snapshot,rotate) #all snapshots are rotated to a standard orientation
        #print("ranges :", np.where(np.isinf(self.current_snapshot)==False))
        #print("current image :",self.current_snapshot)

        landmark_exists = not np.all(np.isinf(self.current_snapshot)) #checking if there is one element in the snapshot array where the range is not inf

        if landmark_exists:

            self.current_landmarks = self.segment_landmarks(self.current_snapshot)
            self.current_gaps  = self.segment_gaps(self.current_snapshot)
            self.current_landmark_vectors = self.slices_to_vectors(self.current_landmarks)
            self.current_gap_vectors = self.slices_to_vectors(self.current_gaps)

            print("Landmarks Slices: ", self.current_landmarks)
            print("Gaps Slices: ", self.current_gaps)
            print("Landmark Vectors : ", self.current_landmark_vectors)
            print("Target Landmark Vectors : ", self.target_landmark_vectors)

            if self.target_snapshot is None:
                self.update_target(None)

            if self.homing:
                self.home()
                

        self.rate.sleep()

    def update_target(self,req):
        self.target_snapshot = self.current_snapshot
        self.target_gap_vectors = self.current_gap_vectors
        self.target_landmark_vectors = self.current_landmark_vectors

    def slices_to_vectors(self, sectors):
        vectors = []
        for sector in sectors:
            left_edge, right_edge = sector
            width = right_edge-left_edge #width of the apparent landmark

            if width<0:
                width+=HORIZON_SLICES #if a landmark wraps over the right edge to the beginning of the array, we pretend the array extends further
            angle = self.angles[left_edge]+(self.increment*width/2.0)
            angle %= np.pi*2

            vectors.append((width,angle))

        return vectors


    def segment_landmarks(self, ranges, edge_vectors=False):
        landmarks = []
        left_edge = float("nan") #this is the position of the left edge on the retinal image
        #print("ranges : ", ranges)
        #If we take the center of objects as landmarks
        for i in range(HORIZON_SLICES):
            if(np.isnan(left_edge)): #if we are not on a landmark
                if not np.isinf(ranges[i]): #if we encounter a landmark
                    left_edge = i
                    print("Left edge at : ", i, " dist  :", ranges[i])
            #if we are on a landmark        
            else:
                if np.isinf(ranges[i]): #and if we hit the edge, we close the landmark and append it 
                    #print("Segment [%s,%s]"%(left_edge,i-1))
                    landmarks.append((left_edge,i-1))
                    left_edge = float("nan")

        #Below we check whether there is a landmark warping over the the scan array
        if not np.isnan(left_edge):
            if not np.isinf(ranges[i]):
                _, e = landmarks[0]
                landmarks[0] = (left_edge,e)
            else:
                landmarks.append(left_edge,i-1)
        
        return landmarks

    def segment_gaps(self, ranges):
        gaps = []
        #ranges = np.where(np.isinf(ranges))
        left_edge = float("nan") #this is the position of the left edge on the retinal image
        print("ranges : ", ranges)

        #If we take the center of objects as landmarks
        for i in range(HORIZON_SLICES):
            if(np.isnan(left_edge)): #if we are not on a gap
                if np.isinf(ranges[i]): #if we encounter a gap
                    left_edge = i
            #if we are on a gap...
            else:
                if not np.isinf(ranges[i]): #...and encounter a landmark
                    #print("Segment [%s,%s]"%(left_edge,i-1))
                    gaps.append((left_edge,i-1))
                    left_edge = float("nan")

        #Below we check whether there is a gap warping over the the scan array
        if not np.isnan(left_edge):
            if np.isinf(ranges[i]):
                _, e = gaps[0]
                gaps[0] = (left_edge,e)
            else:
                gaps.append(left_edge,i-1)
        return gaps

    def computeHomingVector(self, current_vectors,target_vectors):
        '''Computes the home vector.
        
        @param current_vectors: vectors to segments in the current retinal image.
        
        @param target_vectors: vectors to segments in the target retinal image.'''

        n_target_vectors = len(target_vectors)
        neighbours = np.zeros(n_target_vectors)

        vectors = np.zeros((n_target_vectors,2)) #for each sector, there is a radial and a tangential vector
        radial_vectors = np.zeros((n_target_vectors,2))
        tangential_vectors = np.zeros((n_target_vectors,2))

        for i in range(n_target_vectors):
            closest_distance = float("inf")

            target_r, target_theta = target_vectors[i]

            for j in range(len(current_vectors)):
                current_r, current_theta = current_vectors[j]
                alpha = target_theta-current_theta
                alpha = ((alpha+np.pi)%(np.pi*2))-np.pi #Gives the shortest angle between two points on the unit circle, so we don't make a full turn to go from 180 degrees to 179
                
                rospy.logdebug("From home %s to current %s, alpha: %s"%(i,j,alpha))

                if np.abs(alpha)<np.abs(closest_distance):

                    closest_distance = alpha
                    neighbours[i] = j
                    delta_r = target_r-current_r
                    vectors[i] = [delta_r,alpha]
            
            rospy.logdebug("From home landmark %s to closest current landmark %s, r: %s, t: %s"%(i, neighbours[i], delta_r, closest_distance))
                    

            radial_vectors[i] = cartesian((delta_r,target_theta))#radial vector
            tangential_vectors[i] = cartesian((closest_distance,target_theta+np.pi/2))#tangential vector

        home_vector = np.sum((radial_vectors*RADIAL_VECTOR_WEIGHT+tangential_vectors*TANGENTIAL_VECTOR_WEIGHT)/(RADIAL_VECTOR_WEIGHT+TANGENTIAL_VECTOR_WEIGHT),axis=0)

        return home_vector


    def home(self):
        '''Returns the agent to the target (i.e. nest) location. Publishes to /cmd_vel topic.'''
        home_vector = self.computeHomingVector(self.current_landmark_vectors, self.target_landmark_vectors) 
        home_vector += self.computeHomingVector(self.current_gap_vectors, self.target_gap_vectors) 
        home_vector = home_vector/2

        r,alpha = polar((home_vector[0],home_vector[1]))

        print("Current home vector :", home_vector[0], home_vector[1]-self.yaw)

        slow_down = np.pi/4 #If the difference between the target and the current angles is smaller than this, the turning rate will be adjusted proportionally
        
        alpha = alpha - self.yaw #relative angle of the home vector
        alpha = ((alpha+np.pi)%(np.pi*2))-np.pi #Gives the shortest angle between two points on the unit circle, so we don't make a full turn to go from 180 degrees to 179
        
        #Adjust the vector amplitude
        r/=5 #arbitrary scaling factor
        r = np.clip(r,-1,1) #clip between [-1,1]

        #Assume the agent's at the nest and stop if the vector length is too little.
        if np.abs(r)<0.2:
            r = 0

        z = np.clip(alpha/slow_down,-1,1)
        
        msg = Twist()
        msg.angular.z = z*MAXIMUM_ANGULAR_SPEED
        msg.linear.x = r*np.cos(alpha)*MAXIMUM_LINEAR_SPEED
        self.pub.publish(msg)

    def imu_callback(self, msg):
        # Convert quaternions to Euler angles.
        (_,_,self.yaw) = tf.transformations.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def halt(self):
        msg = Twist()
        msg.angular.z = 0
        msg.linear.x = 0
        self.pub.publish(msg)

def cartesian(vector):
    r,theta = vector
    return np.array([r*np.cos(theta),r*np.sin(theta)])

def polar(vector):
    x,y = vector
    return np.array([np.sqrt(x**2+y**2),np.arctan2(y,x)])

if __name__ == '__main__':
    Snapshot()

    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()
        print("Killing node...")