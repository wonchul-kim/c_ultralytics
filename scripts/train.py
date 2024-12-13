from ultralytics import YOLO


model = YOLO("/HDD/weights/yolov11/yolo11n.pt")
train_results = model.train(
    data="tmp.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=1024,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    label_format='labelme'
)
