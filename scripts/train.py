from ultralytics import YOLO, RTDETR, settings
from callbacks.aivdb import *
settings.update({'wandb': False})


model = YOLO("/HDD/weights/yolov11/yolo11x.pt")
# model.add_callback('on_train_start', on_train_start)
# model.add_callback('on_train_epoch_end', on_train_epoch_end)
# model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
# model.add_callback('on_model_save', on_model_save)
# model.add_callback('on_train_end', on_train_end)


# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=80,  # training image size
#     device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     label_format='yolo'
# )

train_results = model.train(
    data="/DeepLearning/etc/_athena_tests/recipes/agent/detection/pytorch/train/rectangle2/rectangle2.yaml",  # path to dataset YAML
    epochs=300,  # number of training epochs
    imgsz=1024,  # training image size
    batch=16,
    device="0,1,2,3",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    label_format='yolo',
    lr0=0.005,
    lrf=0.001,
    optimizer='SGD'
)

# train_results = model.train(
#     data={'path': '/DeepLearning/_athena_tests/datasets/rectangle2/split_yolo_hbb_dataset',
#           'train': 'train',
#           'val': 'val',
#           'names': {0: 'RING', 1: 'DUST', 2: 'SCRATCH', 3: 'FOLD', 4: 'DAMAGE', 5: 'LINE', 6: 'BOLD', 7: 'BURR', 8: 'BUBBLE', 9: 'TIP', 10: 'REACT'}},
#     epochs=300,  # number of training epochs
#     imgsz=1024,  # training image size
#     batch=16,
#     device="0,1,2,3",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
#     label_format='labelme',
#     lrf=0.001,
# )

