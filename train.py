import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('ultralytics/cfg/models/v8/yolov8m-traffic.yaml') # loading pretrain weights
    
    model.train(data='data.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=20,
                workers=4,
                device='0',
                optimizer='SGD', # using SGD
                project='runs/train',
                name='exp'
                )