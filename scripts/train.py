from ultralytics import YOLO


model = YOLO("/HDD/weights/yolov11/yolo11n.pt")
train_results = model.train(
    data="coco128.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=80,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    label_format='yolo'
)

# train_results = model.train(
#     data="tmp.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=1024,  # training image size
#     device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     label_format='labelme'
# )

# train_results = model.train(
#     epochs=100,  # number of training epochs
#     imgsz=1024,  # training image size
#     device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     label_format='labelme',
#     data={'path': '/DeepLearning/_athena_tests/datasets/rectangle1/split_dataset_unit',
#           'train': 'train',
#           'val': 'val',
#           'names': {0: 'NUMBER_OK', 1: 'NUMBER_NG', 2: 'LOT_OK', 3: 'LOT_NG' }},
#     roi_info = [[0, 0, 2448, 2048]],
#     roi_from_json = False, 
# )
