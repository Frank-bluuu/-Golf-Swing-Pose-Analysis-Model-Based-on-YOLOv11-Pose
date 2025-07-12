from ultralytics import YOLO


# Load a model
model = YOLO(r"/home/linux/Project/YOLO/ultralytics-main/ultralytics/cfg/models/11/yolo11s-pose.yaml")  # build a new model from YAML
model = YOLO(r"/home/linux/Project/YOLO/ultralytics-main/yolo11s-pose.pt") # load a pretrained model

# Predict with the model
results = model.train(data=r'datasets/golf.yaml',
                      imgsz=640,
                      batch=4,
                      epochs=150,
                      device=0,
                      workers=8,
                      cache=False,
                      amp=True)