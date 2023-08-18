from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
#model.to('cuda')

# Training.
results = model.train(data='custom_data.yaml',imgsz=5,epochs=10,batch=8,name='yolov8n_custom')

