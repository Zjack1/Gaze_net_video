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
import copy
loss = nn.L1Loss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt


def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi


if __name__ == "__main__":
  config = yaml.load(open("config.yaml"), Loader = yaml.FullLoader)
  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])

  dataset = reader_test.txtload(labelpath, imagepath, 1500, shuffle=False, num_workers=0, header=True)


  net = model_mobilenetv2.model()
  statedict = torch.load("./Iter_5_GazeNet.pt")

  net.to(device)
  net.load_state_dict(statedict)
  net.eval()

  length = len(dataset)
  accs = 0
  count = 0
  for j, (data, label, img_raw) in enumerate(dataset):
    data["right_img"] = data["right_img"].to(device)
    gts = label.to(device)
    gazes = net(data)
    mae_loss = loss(gazes, label)
    print(mae_loss*180/3.14159)



  '''
    for j, (data, label) in enumerate(dataset):
    data["right_img"] = data["right_img"].to(device)
    gts = label.to(device)
    gazes = net(data)
    l = gazes[0] - label[0]
    l = l*(180/3.14159)
    l = abs(l[0]) + abs(l[1])
    accs = accs + float(l)
    #print(l)
    # mae_loss = loss(gazes, label)
    # print(mae_loss*180/3.14159)
  print("loss ",accs/1500)
  '''
