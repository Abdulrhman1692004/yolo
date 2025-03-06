from ultralytics import YOLO
import cv2 as cv
import math
import numpy as np
from sort import *

class Detect:
    def __init__(self, video_source, model_path):
        """Initialize the YOLO model and video source."""
        self.model = YOLO(model_path)
        self.video_capture = cv.VideoCapture(video_source)  # FIXED: Properly initializing VideoCapture
        self.class_names = self.load_class_names()
        self.mot_tracker = Sort(max_age= 20, min_hits=2, iou_threshold=0.3) 
        #max_age: the limit of the number of frames that it is gone and we still recogize it 
        #iou_threshold: to find how get the intersection of overlapping boxes is


    def load_class_names(self):
        """Loads the class names for YOLO detection."""
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def process_frame(self, frame):
        """Processes a single frame and applies YOLO object detection."""
        results = self.model(frame, stream=True)
        detections = np.empty((0, 5))
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = (int((x1 + x2)/2),int((y1 + y2)/2) )
                radius = int((math.sqrt((x2 - x1)**2 + (y2 - y1)**2))/2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if(conf >= 0.53 and cls == 2): # 2 is the class for cars
                    #cv.circle(frame, center, radius, (0, 255, 0),0)
                    current_Array = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections,current_Array))
                    tracker_results = self.mot_tracker.update(detections)
                    '''
                    [x1,y1,x2,y2,ID]
                    tracker_results = np.array([[100, 50, 150, 100, 1],  # Tracked object 1 with ID 1
                            [200, 80, 250, 130, 2]])  # Tracked object 2 with ID 2
                    '''


                    for tracked in tracker_results:
                        X1,Y1,X2,Y2,ID = tracked
                        
                        cv.putText(frame, f'{self.class_names[cls]}{int(ID)}',
                         (int(X1), int(Y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (10, 3, 7), 1)

        return frame

    def run(self):
        """Runs the object detection on the video feed."""
        while True:
            is_true, frame = self.video_capture.read()  # FIXED: Using `self.video_capture`
            if not is_true:
                break

            frame = self.process_frame(frame)
            cv.imshow('YOLO Object Detection', frame)

            if cv.waitKey(20) & 0xFF == ord('d'):
                break

        self.video_capture.release()
        cv.destroyAllWindows()


# Running the YOLO Object Detection
detector = Detect('cars.mp4', 'yolov8n.pt')
detector.run()
