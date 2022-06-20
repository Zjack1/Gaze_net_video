import cv2
import argparse
import numpy as np
import torch
from torchvision.transforms import transforms
from UNet import Unet

x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5])])  # ->[-1,1]

model_path = "./saved_model/UNet_32_bruce_20.pth"
y_transforms = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Unet(1, 1).to(device)
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint)
model = net.eval()


def unet(eye_roi_img_raw, draw_img, x1, y1):
    eye_roi_img_w = eye_roi_img_raw.shape[1]
    eye_roi_img_h = eye_roi_img_raw.shape[0]
    eye_roi_gray = cv2.cvtColor(eye_roi_img_raw, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(eye_roi_gray)
    eye_roi_img = cv2.resize(img, (60,36))
    img_x = x_transforms(eye_roi_img)
    img_x = img_x.unsqueeze(0).to(device)
    out_mask = model(img_x)
    predict = out_mask.squeeze(0).data.cpu().numpy()
    predict = predict[0]
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 255
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    predict = predict.astype(np.int16)
    cv2.imwrite("./temp_mask.png", predict)
    mask = cv2.imread('./temp_mask.png', 1)
    mask = cv2.resize(mask, (eye_roi_img_w,eye_roi_img_h))

    imgROIAdd = cv2.add(mask, eye_roi_img_raw)  # 区域图像与mask图像叠加融合
    draw_img[y1:y1+eye_roi_img_h, x1:x1+eye_roi_img_w] = imgROIAdd


    # ret, thresh = cv2.threshold(mask, 111, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(eye_roi_img_raw, contours, -1, (255, 255, 255), 1)
    cv2.imshow('result', draw_img)