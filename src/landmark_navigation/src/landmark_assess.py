#!/usr/bin/env python
import rospy
import numpy as np
import os

from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

THRESHOLD = 2 #Threshold for nest vicinity, meters

#Sound alert, users should install "sox" from the package manager.
duration = 0.3
freq = 440
class LandmarkAssess:
    '''A helper class that records the times, distances and tortuosity of LN modules. '''
    def __init__(self): 
        rospy.init_node("landmark_assess")
	
        self.position = None

        #Agent's steps are recorded to compute the total path length (not the displacement)
        self.last_x = 0
        self.last_y = 0
        self.length = 0

        #Stats for each run
        self.lastDistance = 0 #euclidian distance from the nest
        self.t_begin = 0 #beginning of a home run
        self.time = rospy.get_time() #Gets simulated time

        #Lists to record stats
        self.history_lengths = []
        self.history_durations = []
        self.history_distances = []

        self.count = 0

        #Only Gazebo odometry is needed.
        rospy.Subscriber("/odom", Odometry, self.odometry_callback, queue_size=1)

        self.rate = rospy.Rate(4)

    def odometry_callback(self,data):
        '''ROS Subscriber callback function, tracks the agent and records homing stats. Warns the operator with a beeping sound when the agent arrives at the nest.'''
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y

        dx = x-self.last_x
        dy = y-self.last_y

        self.length+=np.sqrt(dx**2+dy**2)

        distance_from_nest = np.sqrt(x**2+y**2)
        self.time = rospy.get_time()

        #If the nest is reached...
        if distance_from_nest<THRESHOLD and self.lastDistance>THRESHOLD:
            d_t = self.time-self.t_begin
            self.count = self.count + 1
            rospy.loginfo("#%s REACHED THE TARGET in %ss and length: %s"%(self.count,d_t,self.length)) 

            self.history_lengths.append(self.length)
            self.history_durations.append(d_t)

            rospy.loginfo("---------------------")
            #os.system("play -nq -t alsa synth %s  sine %s"%(duration,freq)) #Produce a beeping sound to tell the operator that the agent has reached the nest


        #If the agent was near the nest and "kidnapped" to a new location    
        elif distance_from_nest>THRESHOLD and self.lastDistance<THRESHOLD:
            rospy.loginfo("Repositioned, resetting the chronometer...")
            rospy.loginfo("The agent is %sm far from the nest."%distance_from_nest)
            self.history_distances.append(distance_from_nest)
            self.t_begin = self.time
            self.length = 0

        self.lastDistance = distance_from_nest
        self.last_x = x
        self.last_y = y
        self.rate.sleep()

if __name__ == '__main__':
    #os.system("play -nq -t alsa synth %s  sine %s"%(duration,freq)) #Produce a beeping sound at start.
    LA = LandmarkAssess()
    while not rospy.is_shutdown(): #loop until a shutdown signal is received...
        rospy.spin()
    print("Killing node...")
    np.savez("landmarknavigationstats.npz", h_distance=LA.history_distances,h_durations=LA.history_durations,h_length=LA.history_lengths) #save stats to assess LN performance

