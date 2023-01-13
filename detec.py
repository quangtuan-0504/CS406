import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread(r'C:\Users\TUAN\PycharmProjects\CS331\masks.jpg')
#img=r'C:\Users\TUAN\PycharmProjects\CS331\masks.jpg'
model = torch.hub.load('ultralytics/yolov5', 'custom',path=r'C:\Users\TUAN\PycharmProjects\CS331\Yolov5\yolov5\runs\train\exp3\weights\best.pt')

model.agnostic = True
# this prevents the model from showing multiple overlapping boxes

results = model(img)

print(results)

print(results.render())