from ultralytics import YOLO
import cv2 as cv
import cvzone

class Detect:
    def __init__(self, video_source, model_path):
        """Initialize the YOLO model and video source."""
        self.model = YOLO(model_path)
        self.video_capture = cv.VideoCapture(video_source)  # FIXED: Properly initializing VideoCapture
        self.class_names = self.load_class_names()

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
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                cvzone.cornerRect(frame, bbox)

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                cv.putText(frame, f'{self.class_names[cls]} {conf:.2f}',
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (10, 3, 7), 1)
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
detector = Detect('cars.mp4', 'yolo_basic/yolov8n.pt')
detector.run()
