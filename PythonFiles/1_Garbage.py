from ultralytics import YOLO
import json
from urllib.request import urlopen
import cv2
import cvzone
import math
import time
import firebase_admin
from datetime import datetime
from firebase_admin import credentials, storage
from firebase_admin import firestore
import os

# Constants for distance estimation
KNOWN_DISTANCE = 1.0  # distance from camera to object
KNOWN_HEIGHT = 0.32  # height of the object

# Initialize Firebase Admin SDK for Authentication and Firestore
cred = credentials.Certificate("techsavants-bdf20-firebase-adminsdk-flfvo-d5dce4bf44.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'techsavants-bdf20.appspot.com'
})

# Initialize Firebase Storage
bucket = storage.bucket()

# Initialize Firestore
db = firestore.client()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("Weights/litter.pt")

classNames = ['Gravel','litter']

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        detected_objects = []  # To store detected object data

        for box in boxes:
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Store detected object data
            detected_object = {
                'class': classNames[cls],
                'confidence': conf
            }
            detected_objects.append(detected_object)


            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1


            # Estimate distance based on the bounding box height
            distance = (KNOWN_HEIGHT * img.shape[0]) / (h * KNOWN_DISTANCE)
            distance = round(distance, 2)  # Round to 2 decimal places for better readability
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf > 0.5:  # You can adjust the confidence threshold
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get current date and time
                c_time = datetime.now().strftime("%H:%M:%S")
                c_date = datetime.now().strftime("%d-%m-%Y")

                url = "http://ipinfo.io/json"
                response = urlopen(url)
                data = json.load(response)
                lat, lon = map(float, data['loc'].split(','))

                # Create the directory if it doesn't exist
                os.makedirs("image/garbage_image", exist_ok=True)

                # Rest of your code remains the same
                image_filename = f'image/garbage_image/{timestamp}.jpg'

                # Save the image locally
                cv2.imwrite(image_filename, img)

                # Upload the image to Firebase Storage
                blob = bucket.blob(image_filename)
                blob.upload_from_filename(image_filename)

                # Get the URL of the uploaded image
                image_url = blob.public_url

                # Access Firestore and add the detected_objects data with the image URL
                detected_objects_collection = db.collection("garbage_objects")
                detected_objects_collection.add({
                    'Date': c_date,
                    'Time': c_time,
                    'Issue': classNames[cls],
                    'Confidence': conf,
                    'Lattitude': lat,
                    'Longitude': lon,
                    'timestamp': timestamp,
                    'image_url': image_url
                })

            if distance <= 3.0:
                myColor = (0, 0, 255)  # Red in BGR format
            else:
                myColor = (0, 255, 0)  # Green in BGR format

            # Display distance
            cvzone.putTextRect(img, f'Distance: {distance}m', (x1, y1 - 35), scale=1, thickness=1)

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                               (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                               colorT=(0, 255, 255), colorR=myColor, offset=5)
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Image", img)
    cv2.waitKey(1)