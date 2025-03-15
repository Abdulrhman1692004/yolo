import cv2 as cv
import numpy as np

video = cv.VideoCapture("ballons.mp4")

if not video.isOpened():
    print("Error: Could not open video file")
    exit()

while True:
    is_true, frame = video.read()
    if not is_true:
        break
    
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    frame = cv.resize(frame, (640, 640))
    
    # Convert the frame to HSV color space
    HSV_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Define range for detecting red color in HSV space
    low_red = np.array([161, 155, 60])
    high_red = np.array([179, 255, 255])
    
    # Create a mask for red color
    red_mask = cv.inRange(HSV_img, low_red, high_red)
    
    # Find contours in the mask: Contours are the boundaries of the objects. they serve as to detect clusters of certain objects
    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #needs to be understood)
    
    # Draw bounding boxes around the detected red objects
    for contour in contours:
        # If the contour is large enough (to avoid noise), draw a bounding box
        if cv.contourArea(contour) > 10:  # Adjust this threshold based on your objects size
            x, y, w, h = cv.boundingRect(contour) #(NEW)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
    
    # Show the resulting frame with bounding boxes
    cv.imshow("Red Object Detection", frame)
    
    if cv.waitKey(20) & 0xFF == ord('q'):  # Press 'q' to exit
        break

video.release()
cv.destroyAllWindows()
