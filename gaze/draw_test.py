import model_mobilenetv2
import reader_test
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
from math import cos, sin
loss = nn.L1Loss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def draw_eye_line(img, pitch, yaw, color, size=60):
  tdx = 30
  tdy = 18

  # Z-Axis (out of the screen) drawn in blue
  x3 = size * (-sin(yaw)) + tdx
  y3 = size * (-cos(yaw) * sin(pitch)) + tdy

  cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), color, 2)
  return img


if __name__ == "__main__":
  config = yaml.load(open("./config.yaml"), Loader = yaml.FullLoader)
  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])

  dataset = reader_test.txtload(labelpath, imagepath, 1, shuffle=False, num_workers=0, header=True)


  net = model_mobilenetv2.model()
  statedict = torch.load("./Iter_12_GazeNet.pt")

  net.to(device)
  net.load_state_dict(statedict)
  net.eval()

  length = len(dataset)
  accs = 0
  count = 0

  for j, (data, label, img_raw) in enumerate(dataset):
    data["right_img"] = data["right_img"].to(device)
    img_raw = img_raw[0].numpy()
    gazes = net(data)
    img_draw = draw_eye_line(img_raw, float(label[0][0]), float(label[0][1]),(0, 255, 0))
    img_draw = draw_eye_line(img_draw, float(gazes[0][0]), float(gazes[0][1]),(0, 0, 255))
    cv2.imshow("sss",img_draw)
    cv2.waitKey(999)
