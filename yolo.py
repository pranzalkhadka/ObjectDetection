from ultralytics import YOLO
import cv2
import math
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,900)
cap.set(4,600)
#Accessed the webcam and changed the size 

model = YOLO('../yolo-weights/yolov8l.pt')
#Importing the yolo version 8 

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
#Got the coco dataset classes

while True:
    Check,img = cap.read()
    predictions = model(img,stream=True)
    #passed the images to model for predictions
    for preds in predictions:
        bounding_boxes=preds.boxes
        for box in bounding_boxes:
            x1,y1,w,h=box.xyxy[0]
            x1,y1,w,h = int(x1),int(y1),int(w),int(h)
            #Got the starting point,width and height of the bounding box
            cv2.rectangle(img,(x1,y1),(w,h),(255,0,0),3)
            #Formed a rectangle around the image
            conf = math.ceil(box.conf[0]*100)/100
            cls= box.cls[0]
            cls = int(cls)
            #Got the confidence and class label of the displayed image
            cvzone.putTextRect(img,f'{classes[cls]} {conf}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
            #Displaying the class and confidence on top of the bounding box
    cv2.imshow("image",img)
    cv2.waitKey(1)