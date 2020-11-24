# Overview
This project contains the source code that accompanies my final year project. The project consists of experiments assessing various Path Integration and Landmark-based Navigation techniques, varying in biological plausibility. The report aims to find a biologically plausible methods that have low-overhead and are possibly implementable on analog robots, like the _Central Complex_ and _Average Landmark Vector_ models.

Path Integrator models are assessed based on their robustness against noise and precision, whereas Landmark-based Navigation models are assessed on their accuracy of homing after the PI module is zeroed (i.e. when the agent falsely thinks it's home), robustness against occlusion, and emergent behaviours like obstacle avoidance, if any.

## SETUP:

In order to setup the modules, the following steps should be followed (assuming your workspace is on `~/catkin_ws`):

* ROS Kinetic should be installed following the instructions here: http://wiki.ros.org/kinetic/Installation
* Turtlebot ROS packages should be installed: http://emanual.robotis.com/docs/en/platform/turtlebot3/setup/#setup
* The contents of the `model` directory should be moved under `~/.gazebo/models` directory.
* The XACRO files in `xacro_files` directory should be moved under `~/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/`, replacing the existing files.
* The directories inside the `src` directory should be moved under `~/catkin_ws/src/`.

Finally,
* The messages and services be compiled by running `catkin_make` from the terminal inside the `~/catkin_ws` directory.

## RUNNING NODES:

PI and LN nodes can be launched by using roslaunch, this will also bring up a Gazebo world with 3D models and a custom turtlebot3 (except for helper nodes like driver, comparator, landmark_assess, central_complex_plot). 

If you prefer to test the nodes in a custom environment, run `roslaunch gazebo_ros empty_world.launch`, import model files into the Gazebo world, and finally, simply run python files of individual nodes you wish to test (e.g. `python snapshot.py`).


