#! /usr/bin/env python
# -*- coding: utf8 -*-

import rospy
from sensor_msgs.msg import PointCloud2, Image

from visualization_msgs.msg import Marker,MarkerArray

import numpy as np
import ros_numpy as rnp
import cv2
from cv_bridge import CvBridge

import tf2_ros

import glob
import os

from mask_rcnn_ros.srv import DetectObjects

detect_service = None
cv_bridge = None
marker_pub = None

objects_dic = {}

def tf2np(tr):
    from pyquaternion import Quaternion as Q
    ret = np.eye(4,4)
    r = tr.transform.rotation
    t = tr.transform.translation
    R = Q(r.w, r.x, r.y, r.z).rotation_matrix
    T = np.array([t.x, t.y, t.z])

    return R, T

def colorize(xyz, zmin = 0, zmax = 3):
    zrng = zmax - zmin
    tmpz = xyz[:,:,2].copy()
    tmpz = ((tmpz - zmin) / zrng * 255)
    tmpz = np.clip(tmpz, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(tmpz, cv2.COLORMAP_JET)

def pc2rgbxyz(points_msg):
    pc = rnp.numpify(points_msg)
    w = points_msg.width
    h = points_msg.height
    xyz = pc.view('<f4').reshape(h,w,pc.itemsize // 4)[:,:,0:3]
    rgb = pc['rgb'].copy().view('<u1').reshape(h,w,4)[:,:,0:3]

    return rgb, xyz

############################################################
# Przetwarzanie chmury punktów
############################################################

def process(score, class_id, class_name, mask, xyz, trans):

    points = xyz[mask!=0]

    c = np.nanmean(points, axis=0)

    x = points[:,0]
    xmin = min(x)
    xmax = max(x)

    R, T = tf2np(trans)
    # przekształcenie położenia znaku
    c = np.matmul(R, c) + T
    points = np.transpose(np.matmul(R, np.transpose(points))) + T

    z = points[:,2]
    zmin = min(z)
    zmax = max(z)


    rospy.loginfo("Detected %s at %f, %f, %f", class_name, c[0], c[1], c[2])

    marker_ = Marker()
    marker_.header.frame_id = "/odom"
    marker_.type = marker_.CYLINDER
    marker_.action = marker_.ADD

    marker_.pose.position.x = c[0]
    marker_.pose.position.y = c[1]
    marker_.pose.position.z = c[2]
    marker_.pose.orientation.x = 0
    marker_.pose.orientation.y = 0
    marker_.pose.orientation.z = 0
    marker_.pose.orientation.w = 1
  
    marker_.scale.x = xmax-xmin
    marker_.scale.y = xmax-xmin
    marker_.scale.z = zmax-zmin
    marker_.color.a = 0.5
    marker_.color.r = 1.0
    marker_.color.g = 0.0
    marker_.color.b = 0.0

    print(objects_dic)

    # każdy obiekt musi dostać unikalne id!
    found= False
    for key, values in objects_dic.items():
        if np.linalg.norm(values-c)<0.2:
            found = True
            marker_.id= key
            return marker_


    if(not found):
        objects_dic[len(objects_dic)+1]=c
        marker_.id = len(objects_dic)
        
    return marker_
    

def callback(points_msg):
    global detect_service

    while True:
        try:
            trans = tfBuffer.lookup_transform(target_frame, points_msg.header.frame_id, points_msg.header.stamp)
            break
        except:
            print("Retry tf lookup")
            rospy.sleep(0.1)
    
    # zamiana chmury punktów na obraz kolorowy oraz macierz współrzędnych
    rgb, xyz = pc2rgbxyz(points_msg)
    
    try:
        image_msg = cv_bridge.cv2_to_imgmsg(rgb, 'bgr8')
        image_msg.header = points_msg.header
        resp = detect_service(image_msg)
        r = resp.result
        rospy.loginfo("Detected %d objects", len(r.scores))

        marker_array_msg = MarkerArray()

        # process detections
        for sc, ci, cn, ms in zip(r.scores, r.class_ids, r.class_names, r.masks):
            if cn=="bottle":
                mask = cv_bridge.imgmsg_to_cv2(ms, 'mono8')
                #tutaj erozja
                # Creating kernel
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel) 

                marker = process(sc, ci, cn, mask, xyz, trans)
                marker_array_msg.markers.append(marker) 

        marker_pub.publish(marker_array_msg)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    return


if __name__ == '__main__':
    rospy.init_node('symbol_detector', anonymous=True)

    target_frame = rospy.get_param('~target_frame', 'odom')

    # przygotowanie i wstępne wypełnienie bufora transformacji
    tfBuffer = tf2_ros.Buffer(rospy.Duration(120))
    listener = tf2_ros.TransformListener(tfBuffer)

    cv_bridge = CvBridge()

    rospy.loginfo("Filling TF buffer")
    rospy.sleep(2)


    rospy.loginfo("Subscribing to topics")

    marker_pub = rospy.Publisher("/visualization_marker", MarkerArray, queue_size = 2)
    points_sub = rospy.Subscriber('points', PointCloud2, callback, queue_size=1)


    rospy.wait_for_service('detect_objects')
    detect_service = rospy.ServiceProxy('detect_objects', DetectObjects)

    rospy.spin()