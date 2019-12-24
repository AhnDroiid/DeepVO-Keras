import rospy
import numpy as np
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu

import Config
import time
import math

import csv
DR_NETWORK = False
vehicle = Config.Vehicle()

LoamPath = Path()
DrPath = Path()
# Loam_x = 0
# Loam_y = 0
current_velocity = 0


curr_steering = 0
curr_velocity = 0
loam_x = 0
loam_y = 0
forward_v = True
start_vel = None
IMU_pub = rospy.Publisher('Corrected_vel', Float32MultiArray)
def LocalizationCallback(msg):
    global loam_x, loam_y
    loam_x = msg.data[0]
    loam_y = msg.data[1]

def VeloCallback(msg):
    global curr_velocity
    curr_velocity = msg.data[0]*0.278

def SteerCallback(msg):
    global curr_steering
    curr_steering = msg.data[0]

def GearCallback(msg):
    global forward_v
    forward_v = msg.data


def main():

    rospy.init_node('dead_reckoning', anonymous=True)
	
    
    rospy.Subscriber('LocalizationData', Float32MultiArray, LocalizationCallback)

    rospy.Subscriber('CanVelData2', Float32MultiArray, VeloCallback)
    rospy.Subscriber('currentSteer', Float32MultiArray, SteerCallback)
    rospy.Subscriber('forward', Bool, GearCallback)
    #rospy.Subscriber('velocity_INS', Float32MultiArray, IMUCallback)

    loam_path_pub = rospy.Publisher('/LoamPath', Path, queue_size=10)
    dr_path_pub = rospy.Publisher('/DrPath', Path, queue_size=10)

    global LoamPath, DrPath
    global vehicle, curr_steering, curr_velocity, loam_x, loam_y

    LoamPath.header.stamp = rospy.Time.now()
    LoamPath.header.frame_id = "velodyne_link"
    DrPath.header = LoamPath.header

    msg_loam = rospy.wait_for_message('LocalizationData', Float32MultiArray)
    vehicle.x = msg_loam.data[0]
    vehicle.y = msg_loam.data[1]
    vehicle.yaw = 0.0
    # parking_r1: 0.1
    # parking_r2: 2.7
    # parkint_r3: 3.09
    #     
    curr_time = 0
    prev_time = rospy.Time.now()

    while not rospy.is_shutdown():
        try:

            loam_pose = PoseStamped()
            loam_pose.header.stamp = rospy.Time.now()
            loam_pose.header.frame_id = "camera_init"

            dr_pose = PoseStamped()
            dr_pose.header.stamp = rospy.Time.now()
            dr_pose.header.frame_id = "camera_init"   
            loam_pose.pose.position.x = loam_x
            loam_pose.pose.position.y = loam_y
            loam_pose.pose.position.z = 0

            
            curr_time = rospy.Time.now()
            dt = float(str(curr_time - prev_time))*(0.1**9)
            # print(dt)
            prev_time = curr_time

            vehicle.BicycleModel(curr_steering, curr_velocity, dt, forward_v)
            dr_pose.pose.position.x = vehicle.x
            dr_pose.pose.position.y = vehicle.y
            dr_pose.pose.position.z = 0

            LoamPath.poses.append(loam_pose)
            DrPath.poses.append(dr_pose)
            
            loam_path_pub.publish(LoamPath)
            dr_path_pub.publish(DrPath)
            

        except KeyboardInterrupt:
            print("Done.")
            # ms_pub.publish(MsPath)
            break
        
            
if __name__ == '__main__':
    main()