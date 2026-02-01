import cv2
from deepface import DeepFace
from datetime import datetime
import os
import csv

dataset_path = "dataset"
attendance_file = "attendance.csv"
marked_today = set()

# Prepare known faces from dataset
known_faces = []
for filename in os.listdir(dataset_path):
    if filename.lower().endswith((".jpg", ".png")):
        known_faces.append({
            "name": os.path.splitext(filename)[0],
            "path": os.path.join(dataset_path, filename)
        })

# Initialize attendance file
if not os.path.exists(attendance_file):
    with open(attendance_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("📸 Attendance system running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        for person in known_faces:
            result = DeepFace.verify(img1_path=frame, img2_path=person["path"], enforce_detection=False)
            if result["verified"]:
                name = person["name"]
                if name not in marked_today:
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")

                    with open(attendance_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, date, time])

                    print(f"✔️ Marked Present: {name}")
                    marked_today.add(name)
                break  # Stop after first match to speed up

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()