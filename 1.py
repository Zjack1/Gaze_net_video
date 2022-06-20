import cv2
import numpy as np
import matplotlib.pyplot as plt
imgfile = 'face.jpg'
pngfile = 'p14_day01_196_rightlabel.png'

img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 1)
x, y = (110, 110)  # 图像叠加位置
H1, W1 = img.shape[1::-1]
H2, W2 = mask.shape[1::-1]
imgROI = img[y:y + W2, x:x + H2]  # 从背景图像裁剪出叠加区域图像
imgROIAdd = cv2.add(mask, imgROI)  # 区域图像与mask图像叠加融合
#

img[y:y+W2, x:x+H2] = imgROIAdd  # 将融合好的图像还原到原图
cv2.imshow("imgAdd", img)  # 显示叠加图像 imgAdd
cv2.waitKey(0)  # 等待按键命令