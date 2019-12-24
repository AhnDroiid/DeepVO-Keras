import rospy
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image
import numpy as np
import cv2
import message_filters
import rosbag
import argparse
import os
from multiprocessing import Process
from time import sleep, time
pose_topic = "/LocalizationData"
image_topic = "/front_usb_cam/image_raw"
forward_topic = "/forward"
rospy.init_node("Data_Extraction")
data_path = "/data_3/"
image_dat = None
pose_dat = None


#def loam_callback(pose_dat):
image_dat_list = []
pose_dat_list = []
forward_dat_list = []
dt_list = []


def data_processor():
    global image_dat_list, pose_dat_list, forward_dat_list, dt_list
    list_len = len(image_dat_list)
    count = 0
    prev_pose = []
    for idx in range(list_len):
        print("{}/{} is Finished".format(idx, list_len))
        image_dat = image_dat_list[idx]
        pose_dat = pose_dat_list[idx]
        forward_dat = forward_dat_list[idx]
        dt = dt_list[idx]

        image_dat = np.fromstring(image_dat.data, np.uint8).reshape((image_dat.height, image_dat.width, 3))
        image_dat = image_dat[120:280, 110:430, :]
        img_path = os.path.join(os.path.abspath("./") + data_path) + "image{}.png".format(count)
        print(img_path)

        cv2.imwrite(img_path, image_dat)
        line = ""
        line += img_path + " "

        for value in pose_dat.data:
            line += str(value)
            line += " "

        # Append forward topic
        if forward_dat.data:
            line += "true "
        else:
            line += "false "

        # Append dt
        line += str(dt)
        line += " "

        # Append pose change
        if len(prev_pose) == 0:
            for idx, value in enumerate(pose_dat.data):
                if idx == 3: continue
                prev_pose.append(value)
        else:
            for idx, value in enumerate(pose_dat.data):
                if idx == 3: continue
                line += str(value - prev_pose[idx]) + " "
                prev_pose[idx] = value

        # else:
        #     line += str(0.0) + " "
        #     line += str(0.0) + " "
        #     line += str(0.0) + " "

        line += "\n"
        print(line)
        fp = open("." + data_path + "pose_label.txt", "a")
        fp.write(line)
        print("write line")
        count += 1
        fp.close()
    image_dat_list = []
    forward_dat_list = []
    dt_list = []
    pose_dat_list = []


def main():
    global image_dat_list, pose_dat_list, forward_dat_list, dt_list
    first = True
    prev_time = rospy.Time.now()
    count = 0
    while not rospy.is_shutdown():
        try:
            image_dat = rospy.wait_for_message(image_topic, Image, timeout=5)
            pose_dat = rospy.wait_for_message(pose_topic, Float32MultiArray, timeout=5)
            forward_dat = rospy.wait_for_message(forward_topic, Bool, timeout=5)
        except:
            break
        curr_time = rospy.Time.now()
        print("Topic Num: {}".format(count+1))
        if first:
            first = False
            prev_time = curr_time
            continue

        # Get time difference between previous frame and current frame
        dt = float(str(curr_time - prev_time)) * (0.1 ** 9)
        prev_time = curr_time

        image_dat_list.append(image_dat)
        pose_dat_list.append(pose_dat)
        forward_dat_list.append(forward_dat)
        dt_list.append(dt)

        count += 1





if __name__ == "__main__":
    main()
    data_processor()

