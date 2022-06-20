import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch


def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])


class loader(Dataset): 
  def __init__(self, path, root, header=True):

    with open(path) as f:
      self.lines = f.readlines()
    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    img_path = line[0]
    gaze_rad = line[1:3]
    head_rad = line[3:5]

    gaze_rad_label = np.array(gaze_rad).astype("float")
    gaze_rad_label = torch.from_numpy(gaze_rad_label).type(torch.FloatTensor)

    head_rad_label = np.array(head_rad).astype("float")
    head_rad_label = torch.from_numpy(head_rad_label).type(torch.FloatTensor)

    img_raw = cv2.imread(os.path.join(self.root, img_path))
    img = img_raw/255.0
    img = img.transpose(2, 0, 1)

    info = {"right_img": torch.from_numpy(img).type(torch.FloatTensor),
            "head_pose": head_rad_label}

    return info, gaze_rad_label, img_raw


def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = './p00.label'
  d = loader(path)
  print(len(d))
  (data, label) = d.__getitem__(0)

