from ultralytics import YOLO 
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Python Image Sensing Screen Shot App")

img_counter = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab image")
        break

    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    if k%256 == 32: # Spacebar pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("Screenshot taken: {}".format(img_name))
        img_counter += 1

    if k%256 == 27: # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  

# Predict on an image 
for i in range(img_counter):
    img_name = "opencv_frame_{}.png".format(i)
    detection_output = model.predict(img_name, conf=0.25, save=True)
    print("Object detection completed for:", img_name)
