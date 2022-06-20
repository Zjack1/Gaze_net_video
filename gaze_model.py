import model_mobilenetv2
import reader
import numpy as np
import cv2 
import torch
import sys
import yaml
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savepath = "./Iter_10_GazeNet.pt"
net = model_mobilenetv2.model()
statedict = torch.load(savepath)
net.to(device)
net.load_state_dict(statedict)
net.eval()


def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt


def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi


def gaze_estimation(left_eye_roi, euler_angles):
    img = cv2.resize(left_eye_roi, (60,36))
    img = cv2.equalizeHist(img)
    img = np.stack((img,) * 3, axis=-1)

    cv2.imwrite("./eye.jpg", img)
    im_eye = img/255.0
    #im_eye.reshape(36, 60, 3)
    im_eye = im_eye.transpose(2, 0, 1)
    im_eye = torch.tensor([im_eye])
    # im_eye.unsqueeze(1)
    with torch.no_grad():
        headpose = euler_angles[:2]

        input_data = {"eye": torch.tensor(im_eye,dtype=torch.float32).to(device), "head_pose": torch.tensor([headpose],dtype=torch.float32).to(device)}
        gazes = net(input_data)
    return gazes

