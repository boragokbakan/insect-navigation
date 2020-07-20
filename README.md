## SETUP:

In order to setup the modules, the following steps should be followed:

-ROS Kinetic should be installed following the instructions here: http://wiki.ros.org/kinetic/Installation
-Turtlebot ROS packages should be installed: http://emanual.robotis.com/docs/en/platform/turtlebot3/setup/#setup
-The contents of the "model" directory should be moved under "~/.gazebo/models" directory.
-The XACRO files in "xacro_files" directory should be moved under "~/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/", replacing the existing files.
-The directories inside the "src" directory should be moved under "~/catkin_ws/src/".

Finally,
-The messages and services be compiled by running "catkin_make" from the terminal inside the ~/catkin_ws directory.

## RUNNING NODES:

PI and LN nodes can be launched by using roslaunch, this will also bring up a Gazebo world with 3D models and a custom turtlebot3 (except for helper nodes like driver, comparator, landmark_assess, central_complex_plot). 

If you prefer to test the nodes in a custom environment, run "roslaunch gazebo_ros empty_world.launch", import model files into the Gazebo world, and finally, simply run python files of individual nodes you wish to test (e.g. python snapshot.py).


