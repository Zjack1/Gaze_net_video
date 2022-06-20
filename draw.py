#!/usr/bin/env python

import datetime
import logging
import pathlib
from typing import Optional
import math
from math import cos, sin
import cv2
import numpy as np
import MODEL3D
import dlib
from scipy.spatial.transform import Rotation


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


def draw_eye_line(img, pitch, yaw, left_eye_roi_x_mid_coordinate,left_eye_roi_y_mid_coordinate,right_eye_roi_x_mid_coordinate,right_eye_roi_y_mid_coordinate, size = 100):

    pitch_angle = pitch * 180 / np.pi
    yaw_angle = -(yaw * 180 / np.pi)
    s = "eye pitch:" + str(round(pitch_angle,2))+"  eye yaw:"+str(round(yaw_angle,2))
    img = cv2.putText(img, s, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 0, 0), 2)
    print("**************************************************")

    tdx_left = left_eye_roi_x_mid_coordinate
    tdy_left = left_eye_roi_y_mid_coordinate
    x3 = size * (-sin(yaw)) + tdx_left
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy_left
    cv2.line(img, (int(tdx_left), int(tdy_left)), (int(x3), int(y3)), (0, 255, 0), 2)

    tdx_right = right_eye_roi_x_mid_coordinate
    tdy_right = right_eye_roi_y_mid_coordinate
    x3 = size * (-sin(yaw)) + tdx_right
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy_right
    cv2.line(img, (int(tdx_right), int(tdy_right)), (int(x3), int(y3)), (0, 255, 0), 2)

    return img



def draw_new_head_pose_line(img, pitch, yaw, roll, tdx=None, tdy=None, size = 60):
    s = "pitch:" + str(round(pitch,2))+"  yaw:"+str(round(yaw,2))+"  roll:"+str(round(roll,2))
    img = cv2.putText(img, s, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 255, 0), 2)

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = -roll * np.pi / 180
    print("--------------------------------------------")
    #print(pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi)
    #yaw *= -1

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    #cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    #cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def detect_face_box_landmark(image):
    bboxes = detector(image[:, :, ::-1], 0)
    detected = []
    if len(bboxes) == 0:
        return 0
    bbox = bboxes[0]
    predictions = predictor(image[:, :, ::-1], bbox)
    landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],dtype=np.float)
    bbox = np.array([[bbox.left(), bbox.top()], [bbox.right(), bbox.bottom()]],dtype=np.float)
    detected.append(bbox)
    detected.append(landmarks)
    return detected


def draw_face_bbox(image, bbox, color=(0, 255, 0), lw=1):
    assert bbox.shape == (2, 2)
    bbox = np.round(bbox).astype(np.int).tolist()
    cv2.rectangle(image, tuple(bbox[0]), tuple(bbox[1]), color, lw)
    return image


def draw_landmark_points(image, points, color=(0, 255, 255), size=1):
    assert points.shape[1] == 2
    for pt in points:
        pt = tuple(np.round(pt).astype(np.int).tolist())
        cv2.circle(image, pt, size, color, cv2.FILLED)
    return image

def project_points(points3d, rvec, tvec):
    assert points3d.shape[1] == 3
    if rvec is None:
        rvec = np.zeros(3, dtype=np.float)
    if tvec is None:
        tvec = np.zeros(3, dtype=np.float)
    points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        MODEL3D.camera_matrix,
                                        MODEL3D.dist_coefficients)
    return points2d.reshape(-1, 2)


def draw_face_model_axes(image, head_pose_result, landmarks, length=0.05, lw=2):
    AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    # Get the axes of the model coordinate system
    axes3d = np.eye(3, dtype=np.float) @ Rotation.from_euler(
            'XYZ', [0, np.pi, 0]).as_matrix()
    axes3d = axes3d * length
    #a = head_pose_result[0].as_rotvec(degrees=True)
    b = head_pose_result[0].as_rotvec()
    axes2d = project_points(axes3d, b, head_pose_result[1])

    center = landmarks[MODEL3D.NOSE_INDEX]
    center = tuple(np.round(center).astype(np.int).tolist())
    for pt1, color in zip(axes2d, AXIS_COLORS):
        pt1 = tuple(np.round(pt1).astype(np.int).tolist())
        cv2.line(image, center, pt1, color, lw, cv2.LINE_AA)

    return image