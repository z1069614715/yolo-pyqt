import os, cv2, random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import torch
import numpy as np
from yolov7.utils import non_max_suppression, letterbox, scale_coords, plot_one_box

names = ['nomask', 'mask']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

img = cv2.imread('1.jpg')
image, ratio, dwdh = letterbox(img, auto=False)
image = image.transpose((2, 0, 1))[::-1]
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)

im = torch.from_numpy(image).float()
im /= 255
print(im.shape)

model = torch.jit.load(r'weight/yolov7_tiny.pt')
result = model(im)[0]
result = non_max_suppression(result, 0.5, 0.65)[0]
result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)

for *xyxy, conf, cls in result:
    label = f'{names[int(cls)]} {conf:.2f}'
    plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)
    
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()