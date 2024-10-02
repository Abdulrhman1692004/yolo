#importing the libraries
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

#getting a source 
normal_video_capture = cv.VideoCapture('cars.mp4')

#establishing the yolo model object based on its weights
model = YOLO('yolo_basic/yolov8n.pt')
#reading the wieghts 

#getting class names
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]



while True:
    isTrue, frame = webcam_capture.read() 
    #capture.read() returns a list of two arrays [ret, frame]: 
    #   ret: gives if there is a frame or not| | frame: gives the image 
    results = model(frame, stream= True)
    #model intiated before acts as a class upon which when implicated on an image/video produces a set of
    #results recognizing the objects inside it.
    for r in results: #* an example of the results is shown downside
        boxes = r.boxes
        for box in boxes: 
            #x1, y1 ,x2 ,y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = box.xyxy[0]
            # Despite we using xyxy[0], the existence of 4 elements invokes it to get the [0],[1],[2],[3]
            _x,_y,_x2,_y2 = int(x1), int(y1), int(x2), int(y2)
            bbox =  _x,_y,_x2,_y2
            cvzone.cornerRect(frame,bbox)
            
            #get the configiration and the object class_name
            conf = box.conf[0] 
            cls = int(box.cls[0])
            print(conf)
            cv.putText(frame,f'{class_names[cls]}{conf}', (_x, _y - 20),cv.FONT_HERSHEY_SIMPLEX, 1, (10, 3, 7), 1 )

           # cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),3)
    cv.imshow('video', frame) # this shows each and every frame as for showing a video 

    if cv.waitKey(20) and 0xFF ==  ord('d'):
        break
    
webcam_capture.release()
cv.destroyAllWindows()

'''   what gets produced by the results = model() method'''
'''
results = [
    {
        "boxes": [
            {
                "xyxy": [100, 200, 300, 400],  # [x1, y1, x2, y2]
                "conf": [0.85],                # Confidence score
                "cls": [0]                     # Class index for 'person'
            },
            {
                "xyxy": [400, 150, 600, 350],  # [x1, y1, x2, y2]
                "conf": [0.78],                # Confidence score
                "cls": [2]                     # Class index for 'car'
            }
        ]
    }
]


'''
