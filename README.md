
# 利用opencv+yolo v3模型对给定的区域进行物体检测, 若出现违规物体则报警

## request:
    python3
    opencv-python

## install:
    git clone https://github.com/Rainweic/Regional-Object-Detection.git
    cd Regional-Object-Detection/yolo_v3_config
    wget https://pjreddie.com/media/files/yolov3.weights
    wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./yolov3.cfg
    wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./coco.names

## test:
    #Testing the Detection Effect of YOLO V3 With Single Picture
    python3 main.py --image test.jpg