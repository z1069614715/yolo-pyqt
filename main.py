import os, cv2, random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import torch
import numpy as np
from yolo_utils.utils import non_max_suppression, letterbox, scale_coords, plot_one_box
from ultralytics import YOLO, RTDETR

######################################## YOLOV7 ########################################
# names = ['nomask', 'mask']
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# img = cv2.imread('1.jpg')
# image, ratio, dwdh = letterbox(img, auto=False)
# image = image.transpose((2, 0, 1))[::-1]
# image = np.expand_dims(image, 0)
# image = np.ascontiguousarray(image)

# im = torch.from_numpy(image).float()
# im /= 255
# print(im.shape)

# model = torch.jit.load(r'weight/yolov7_tiny.pt')
# result = model(im)[0]
# result = non_max_suppression(result, 0.5, 0.65)[0]
# result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)

# for *xyxy, conf, cls in result:
#     label = f'{names[int(cls)]} {conf:.2f}'
#     plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)
    
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

######################################## YOLOV5 ########################################
# names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# img = cv2.imread('2.jpg')
# image, ratio, dwdh = letterbox(img, new_shape=(960, 960), auto=False)
# image = image.transpose((2, 0, 1))[::-1]
# image = np.expand_dims(image, 0)
# image = np.ascontiguousarray(image)

# im = torch.from_numpy(image).float()
# im /= 255
# print(im.shape)

# model = torch.jit.load(r'weight/yolov5s.pt')
# result = model(im)[0]
# result = non_max_suppression(result, 0.5, 0.65)[0]
# result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)

# for *xyxy, conf, cls in result:
#     label = f'{names[int(cls)]} {conf:.2f}'
#     plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)
    
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

######################################## YOLOV8 ########################################
names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

img = cv2.imread('2.jpg')
model = YOLO('weight/yolov8n.pt')
model.info()

result = next(model.predict(source=img, stream=True, save=False))
result = result.boxes.data.cpu().detach().numpy()
for *xyxy, conf, cls in result:
    label = f'{names[int(cls)]} {conf:.2f}'
    plot_one_box(xyxy, img, label=label, color=colors[int(cls)])

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

######################################## RTDETR ########################################
# names = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#     "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#     "hair drier", "toothbrush"
# ]
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# img = cv2.imread('1.jpg')
# model = RTDETR('weight/rtdetr-l.pt')

# result = next(model.predict(source=img, stream=True, save=False))
# result = result.boxes.data.cpu().detach().numpy()
# for *xyxy, conf, cls in result:
#     label = f'{names[int(cls)]} {conf:.2f}'
#     plot_one_box(xyxy, img, label=label, color=colors[int(cls)])

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()