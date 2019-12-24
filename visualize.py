import matplotlib.pyplot as plt
import cv2
from model import DRVO
import config_dat as cf
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from multiprocessing import Process
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path

LoamPath_pub = None
DRVOPath_pub = None
LoamPath = Path()
DRVOPath = Path()
flag = True

pose_topic = "/LocalizationData"
image_topic = "/front_usb_cam/image_raw"

model = DRVO().model
if cf.weight_path is not None:
    model.load_weights(cf.weight_path)
    print("weight is loaded")
model._make_predict_function()
start_x_DRVO = 0.0
start_y_DRVO = 0.0
start_steering = 0.0
List_SEQ_ImageContainer = []
Numpy_SEQ_ImageContainer = None
Prediction_index = 0

# Bag file execution is needed in console
def loam_callback(pose):
    new_x = pose.data[0]
    new_y = pose.data[1]

    loam_pose = PoseStamped()
    loam_pose.header.stamp = rospy.Time.now()
    loam_pose.header.frame_id = "velodyne_link"
    loam_pose.pose.position.x = new_x
    loam_pose.pose.position.y = new_y
    loam_pose.pose.position.z = 0

    LoamPath.poses.append(loam_pose)
    LoamPath_pub.publish(LoamPath)
    #print("LOAM X: {}, LOAM Y: {}".format(new_x, new_y))



def DRVO_callback(image):
    global List_SEQ_ImageContainer
    global Numpy_SEQ_ImageContainer

    image_dat = np.fromstring(image.data, np.uint8).reshape((image.height, image.width, 3))
    image_dat = image_dat.astype(np.float32) / 255
    image_dat = image_dat[120:280, 110:430, :]
    image_dat -= image_dat.mean()
    image_dat /= image_dat.std()

    # print("Making tensors")
    List_SEQ_ImageContainer.append(image_dat)
    Numpy_SEQ_ImageContainer = np.asarray(List_SEQ_ImageContainer.copy())
    Prediction_loop()


def Prediction_loop():
    global Prediction_index, start_x_DRVO, start_y_DRVO, start_steering, flag
    # print(Numpy_SEQ_ImageContainer.shape[0])
    if  flag and (Numpy_SEQ_ImageContainer.shape[0] - Prediction_index) >= cf.image_stack:
        flag = False
        input_tensor = Numpy_SEQ_ImageContainer[Prediction_index:Prediction_index + cf.image_stack, :, :, :]
        Prediction_index += 1
        input_tensor = np.expand_dims(input_tensor, axis=0)
        # print("Input tensor shape : {}".format(input_tensor.shape))
        # print("SEQ containor shape: {}".format(Numpy_SEQ_ImageContainer.shape))
        print(Prediction_index)
        print(Numpy_SEQ_ImageContainer.shape)
        print("input_tensor shape", input_tensor.shape)
        output_tensor = model.predict(input_tensor, batch_size=cf.image_batch)
        # print(output_tensor)
        # print("Prediction Uploaded")
        # nav msgs PoseStamped
        DRVO_pose = PoseStamped()
        DRVO_pose.header.stamp = rospy.Time.now()
        DRVO_pose.header.frame_id = "camera_init"
        DRVO_pose.pose.position.x = start_x_DRVO + output_tensor[0, 0, 0]
        DRVO_pose.pose.position.y = start_y_DRVO + output_tensor[0, 0, 1]
        DRVO_pose.pose.position.z = 0

        # Append Posestamed msg to Path msg.
        DRVOPath.poses.append(DRVO_pose)
        start_x_DRVO += output_tensor[0, 0, 0]
        start_y_DRVO += output_tensor[0, 0, 1]
        start_steering += output_tensor[0, 0, 2]
        print("Output tensor:", output_tensor)
        print("DRVO x : {}, DRVO y: {}".format(start_x_DRVO, start_y_DRVO))
        DRVOPath_pub.publish(DRVOPath)
        flag = True


def ros_init():
    global LoamPath_pub, DRVOPath_pub
    rospy.init_node("Pose_Prediction")
    DRVO_Sub = rospy.Subscriber(image_topic, queue_size=3, callback=DRVO_callback, data_class=Image)
    Loam_Sub = rospy.Subscriber(pose_topic, queue_size=3, callback=loam_callback, data_class=Float32MultiArray)
    LoamPath_pub = rospy.Publisher('/LoamPath', Path, queue_size=10)
    DRVOPath_pub = rospy.Publisher('/DrPath', Path, queue_size=10)
    LoamPath.header.stamp = rospy.Time.now()
    LoamPath.header.frame_id = "velodyne_link"
    DRVOPath.header = LoamPath.header
    rospy.spin()


ros_init()







