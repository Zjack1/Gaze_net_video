#!/usr/bin/env python
import cv2
import numpy as np
import draw
import MODEL3D
import copy
import torch
#import gaze_model
import gaze.model_mobilenetv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = gaze.model_mobilenetv2.model()
statedict = torch.load("./gaze/Iter_5_GazeNet.pt")
net.to(device)
net.load_state_dict(statedict)
net.eval()
import unet_model

def eye_roi(img, draw_img, landmark):
    th = 0.5  # roi裁剪扩充比例
    left_roi_x1 = min(landmark[41][0], landmark[36][0], landmark[37][0], landmark[38][0], landmark[39][0], landmark[40][0])
    left_roi_y1 = min(landmark[41][1], landmark[36][1], landmark[37][1], landmark[38][1], landmark[39][1], landmark[40][1])
    left_roi_x2 = max(landmark[41][0], landmark[36][0], landmark[37][0], landmark[38][0], landmark[39][0], landmark[40][0])
    left_roi_y2 = max(landmark[41][1], landmark[36][1], landmark[37][1], landmark[38][1], landmark[39][1], landmark[40][1])
    left_eye_roi_x_mid_coordinate = int((left_roi_x2 + left_roi_x1)/2)
    left_eye_roi_y_mid_coordinate = int((left_roi_y2 + left_roi_y1)/2)
    left_eye_w = int(left_roi_x2 - left_roi_x1)
    left_eye_h = int(left_roi_y2 - left_roi_y1)
    th_w = int(left_eye_w * 0.2)
    th_h = int(left_eye_h * 0.5)
    left_eye_roi = img[int(left_roi_y1)-th_h:int(left_roi_y2)+th_h, int(left_roi_x1)-th_w:int(left_roi_x2)+th_w]
    left_eye_roi_x1_coordinate = int(left_roi_x1)-th_w
    left_eye_roi_y1_coordinate = int(left_roi_y1)-th_h
    cv2.rectangle(draw_img, (int(left_roi_x1)-th_w, int(left_roi_y1)-th_h), (int(left_roi_x2)+th_w, int(left_roi_y2)+th_h), (255,0,0), 1)

    right_roi_x1 = min(landmark[42][0], landmark[43][0], landmark[44][0], landmark[45][0], landmark[46][0], landmark[47][0])
    right_roi_y1 = min(landmark[42][1], landmark[43][1], landmark[44][1], landmark[45][1], landmark[46][1], landmark[47][1])
    right_roi_x2 = max(landmark[42][0], landmark[43][0], landmark[44][0], landmark[45][0], landmark[46][0], landmark[47][0])
    right_roi_y2 = max(landmark[42][1], landmark[43][1], landmark[44][1], landmark[45][1], landmark[46][1], landmark[47][1])
    right_eye_roi_x_mid_coordinate = int((right_roi_x2 + right_roi_x1)/2)
    right_eye_roi_y_mid_coordinate = int((right_roi_y2 + right_roi_y1)/2)
    right_eye_w = int(right_roi_x2 - right_roi_x1)
    right_eye_h = int(right_roi_y2 - right_roi_y1)
    th_w = int(right_eye_w * 0.2)
    th_h = int(right_eye_h * 0.5)
    right_eye_roi = img[int(right_roi_y1)-th_h:int(right_roi_y2)+th_h, int(right_roi_x1)-th_w:int(right_roi_x2)+th_w]
    cv2.rectangle(draw_img, (int(right_roi_x1)-th_w, int(right_roi_y1)-th_h), (int(right_roi_x2)+th_w, int(right_roi_y2)+th_h), (0,255,0), 1)

    left_eye_roi_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
    right_eye_roi_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)

    unet_model.unet(left_eye_roi, draw_img, left_eye_roi_x1_coordinate, left_eye_roi_y1_coordinate)
    cv2.imshow("right eye", right_eye_roi_gray)
    cv2.imshow("left eye", left_eye_roi_gray)

    return left_eye_roi_gray, right_eye_roi_gray, draw_img, left_eye_roi_x_mid_coordinate,\
           left_eye_roi_y_mid_coordinate, right_eye_roi_x_mid_coordinate, right_eye_roi_y_mid_coordinate


def gaze_inference_model(eye_roi):
    cv2.imwrite("./eye1.jpg", eye_roi)

    img = cv2.resize(eye_roi, (60, 36))
    img = cv2.equalizeHist(img)
    img = np.stack((img,) * 3, axis=-1)

    cv2.imwrite("./eye.jpg", img)
    im_eye = img/255.0
    #im_eye.reshape(36, 60, 3)
    im_eye = im_eye.transpose(2, 0, 1)
    im_eye = torch.tensor([im_eye],dtype=torch.float32).to(device)
    gazes = net(im_eye)
    return gazes


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        #cv2.imwrite("./face.jpg", frame)
        raw_frame = copy.deepcopy(frame)
        if not ok:
            break
        detected = draw.detect_face_box_landmark(frame)
        if detected == 0:
            continue


        # frame = cv2.imread("./face.jpg")
        # raw_frame = copy.deepcopy(frame)
        detected = draw.detect_face_box_landmark(frame)


        left_eye_roi, right_eye_roi, draw_img, left_eye_roi_x_mid_coordinate,left_eye_roi_y_mid_coordinate,\
        right_eye_roi_x_mid_coordinate, right_eye_roi_y_mid_coordinate = eye_roi(raw_frame, frame, detected[1])

        draw_img = draw.draw_face_bbox(draw_img, detected[0])
        draw_img = draw.draw_landmark_points(draw_img, detected[1])
        head_pose_result = MODEL3D.estimate_head_pose(detected[1])  # 通过2d人脸特征点计算出3d->2d的Rt和人脸姿态的rpy(弧度或者角度)
        # MODEL3D.compute_3d_pose(head_pose_result)  # 通过Rt（人脸姿态的rpy）计算出人脸3d坐标
        # MODEL3D.compute_face_eye_centers(detected[1])  # 计算出左右眼中心位置与人脸中心位置（3d）
        euler_angles = head_pose_result[0].as_euler('XYZ', degrees=True)
        euler_angles_rad = head_pose_result[0].as_euler('XYZ')

        pitch_angle, yaw_angle, roll_angle = euler_angles * np.array([-1, 1, -1])
        draw_img = draw.draw_new_head_pose_line(draw_img, pitch_angle, yaw_angle, roll_angle)
        draw_img = draw.draw_face_model_axes(draw_img, head_pose_result, detected[1])
        print(euler_angles)

        gaze = gaze_inference_model(right_eye_roi)
        print(gaze)
        draw_img = draw.draw_eye_line(draw_img, float(gaze[0][0]), float(gaze[0][1]),left_eye_roi_x_mid_coordinate,left_eye_roi_y_mid_coordinate,right_eye_roi_x_mid_coordinate,right_eye_roi_y_mid_coordinate)


        cv2.imshow('raw', draw_img)
        cv2.waitKey(2)


if __name__ == '__main__':
    main()
