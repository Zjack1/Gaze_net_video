import model_mobilenetv2
import reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import copy
import yaml
import torch.backends.cudnn as cudnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savepath = "./"


if __name__ == "__main__":
  config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
  config = config["train"]
  cudnn.benchmark = True

  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["save"]["model_name"]
  dataset = reader.txtload(labelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=0, header=True)
  net = model_mobilenetv2.model()
  net.train()
  net.to(device)
  # net.load_state_dict(torch.load("./Iter_10_GazeNet.pt"))

  lossfunc = config["params"]["loss"]
  loss_op = getattr(nn, lossfunc)().cuda()
  base_lr = config["params"]["lr"]

  decaysteps = config["params"]["decay_step"]
  decayratio = config["params"]["decay"]

  optimizer = optim.Adam(net.parameters(), lr=base_lr, betas=(0.9, 0.95))
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

  print("Traning")
  length = len(dataset)
  total = length * config["params"]["epoch"]
  cur = 0
  timebegin = time.time()
  for epoch in range(1, config["params"]["epoch"] + 1):
    print("-------epoch:-------",epoch)
    for i, (data, label) in enumerate(dataset):
      data["right_img"] = data["right_img"].to(device)
      #data['head_pose'] = data['head_pose'].to(device)
      label = label.to(device)

      # forward
      gaze = net(data)

      # loss calculation
      loss = loss_op(gaze, label)
      optimizer.zero_grad()

      # backward
      loss.backward()
      optimizer.step()
      scheduler.step()
      cur += 1
      print("epoch:", epoch, "item:", i, "loss: ", loss.item())
    if epoch % config["save"]["step"] == 0:
      torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))


