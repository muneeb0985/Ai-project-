import cv2
import os
name = input("Enter your name: ")
dataset_path = "dataset"
person_path = os.path.join(dataset_path, name)
os.makedirs(person_path, exist_ok=True)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        file_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(file_path, face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow("Collecting Faces", frame)
    if count >= 50:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
