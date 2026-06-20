import cv2
import numpy as np
from ultralytics import YOLO
model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture("video01.mp4")
parking_slots=[
    (100, 200, 200, 350),
    (220, 200, 320, 350),
    (340, 200, 440, 350),
    (460, 200, 560, 350),
]
def is_occupied(slot,boxes):
    x1,y1,x2,y2=slot
    for box in boxes:
        bx1,by1,bx2,by2=box
        if not (bx2<x1 or bx1>x2 or by2<y1 or by1>y2):
            return True
    return False
while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame)
    boxes=[]
    for r in results:
        for box in r.boxes:
            cls=int(box.cls[0])
            if cls in [2,3,5,7]:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                boxes.append((x1,y1,x2,y2))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        for slot in parking_slots:
            x1,y1,x2,y2=slot
            if is_occupied(slot,boxes):
                color=(0,0,255)
                label="Occupied"
            else:
                color=(0,255,0)
                label="Free"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cv2.imshow("Parking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()