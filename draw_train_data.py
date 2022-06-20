import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
from math import cos, sin


def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])


def draw_eye_line(img, yaw, pitch, size=60):
    tdx = 30
    tdy = 18

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (-sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (0, 255, 0), 2)

    return img


def loader(path):
  with open(path) as f:
    lines = f.readlines()
  for idx in range(len(lines)):
    line = lines[idx]
    line = line.strip().split(" ")
    name = line[1]
    gaze2d = line[5]
    eye = line[0]
    temp = line[2]

    if temp == "left":
        gazepose = np.array(gaze2d.split(",")).astype("float")
        root = r"C:\Users\94781\Desktop\MPIIGaze-new\Image"
        img = cv2.imread(os.path.join(root, eye))
        img_draw = draw_eye_line(img, gazepose[1], gazepose[0])
        print(name)
        img_path = "../MPIIGaze-new/draw_p00/" + str(name)
        cv2.imwrite(img_path, img_draw)
        cv2.imshow('raw', img_draw)
        if cv2.waitKey(1) in [ord('q'), 27]:
            break



if __name__ == "__main__":
  path = '../MPIIGaze-new/Label/1.label'
  loader(path)

