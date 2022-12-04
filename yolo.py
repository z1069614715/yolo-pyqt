import os, random, yaml
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import cv2
import torch
import numpy as np
from yolov7.utils import non_max_suppression, letterbox, scale_coords, plot_one_box

class yolov7:
    def __init__(self, model_path, iou_thres, conf_thres, device, names):
        device = self.select_device(device)
        print(device)
        if model_path.endswith('pt'):
            model = torch.jit.load(model_path).to(device)
        elif model_path.endswith('onnx'):
            try:
                import onnxruntime as ort
            except:
                raise 'please install onnxruntime.'
            providers = ['CUDAExecutionProvider'] if device.type != 'cpu' else ['CPUExecutionProvider']
            model = ort.InferenceSession(model_path, providers=providers)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        im = self.processing(image)
        
        if self.model_path.endswith('pt'):
            result = self.model(im)[0]
        elif self.model_path.endswith('onnx'):
            result = self.model.run([i.name for i in self.model.get_outputs()], {'images':im})[0]
        image = self.post_processing(result, im, image)
        
        return image
        
    def processing(self, img):
        image = letterbox(img, auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        image = np.array(image, dtype=np.float32)
        image /= 255
        
        if self.model_path.endswith('pt'):
            im = torch.from_numpy(image).float().to(self.device)
        elif self.model_path.endswith('onnx'):
            im = image
        return im
    
    def post_processing(self, result, im=None, img=None):
        if self.model_path.endswith('pt'):
            result = non_max_suppression(result, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        elif self.model_path.endswith('onnx'):
            result = result[:, 1:]
            ratio, dwdh = letterbox(img, auto=False)[1:]
            result[:, :4] -= np.array(dwdh * 2)
            result[:, :4] /= ratio
            
            for *xyxy, cls, conf in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        
        return img
    
    def select_device(self, device):
        if device == -1:
            return torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
            device = torch.device('cuda:0')
        return device

if __name__ == '__main__':
    # read cfg
    with open('model.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov7(**cfg)
    image_path = '1.jpg'
    # inference
    image = yolo(image_path)
    cv2.imshow('pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()