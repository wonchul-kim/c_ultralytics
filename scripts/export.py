from ultralytics import YOLO, settings
from callbacks.aivdb import *
settings.update({'wandb': False})

format = 'onnx'
batch = 1
opset = 14
workspace = 8
device = 'cuda'
weights_file = "/HDD/weights/yolov11/yolo11n.pt"
output_dir = '/HDD/etc/outputs/ultralytics'

model = YOLO(weights_file)
model.add_callback('on_export_start', on_export_start)
model.export(format=format, batch=batch, opset=opset, 
             workspace=workspace, device=device, output_dir=output_dir)