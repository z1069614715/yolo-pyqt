# Yolo-Pyqt5

yolo with pyqt implement in pytorch.  
2023.4.18 support yolov5!!!  
2024.6.12 support yolov8、rtdetr!!!

#### yolov5、yolov7使用教学
视频教学地址: https://www.bilibili.com/video/BV1BP4y1X7ae/?vd_source=c8452371e7ca510979593165c8d7ac27  
博客地址: https://blog.csdn.net/qq_37706472

#### yolov8、rtdetr使用教学
视频教学地址: 

### untitled.ui -> pyqt5 ui文件  
### gui.py -> 由untitled.ui转换出来的py文件
### run_gui.py -> 运行gui主程序
### main.py -> 调试程序
### yolo.py -> yolo模块代码
### track_utils -> 存放跟踪的代码
### yolo_utils -> 存放关于yolo模型函数的代码

# 注意事项
1. 最好用最新版的去训练得到模型来到这个项目下进行使用,在网上的有些代码是版本比较旧的,不一定可以通用.
2. yolov5版本是基于v7.0.

# 安装教程
1. conda install -c conda-forge lap  
2. https://github.com/samson-wang/cython_bbox 源码安装

# 更新日志
### 2023.4.10
1. 修正选择路径和选择视频的bug.
2. 修正中文路径下没法保存图片的bug.

### 2023.4.18
1. 支持yolov5的torchscript,onnx.(有一个示例模型基于yolov5s,imgs:960x960,dataset:visdrone2019)
2. 修改选择模型的逻辑，更改为选择yaml配置文件，具体模型的路径可以在yaml配置文件中进行选择.
3. 修正停止检测按钮没有正常工作的bug.
4. 配置文件新增imgsz参数用于指定输入大小.
5. 配置文件新增model_type参数用于指定模型种类.
6. 日志信息输出增加统计目标数量.
7. 增加目标跟踪功能,目前支持bytetrack跟踪器.
8. 增加推理时间显示,增加视频检测中的fps显示.

### 2024.6.12
1. 支持yolov8、rtdetr.

# 参考
https://github.com/ultralytics/yolov5  
https://github.com/WongKinYiu/yolov7  
https://github.com/samson-wang/cython_bbox  
https://github.com/ifzhang/ByteTrack  
https://github.com/ultralytics/ultralytics  