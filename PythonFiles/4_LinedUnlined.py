import cv2
import os
from datetime import datetime
from urllib.request import urlopen
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage, firestore
from ultralytics import YOLO
import json

# Initialize Firebase Admin SDK for Authentication and Firestore
cred = credentials.Certificate("techsavants-bdf20-firebase-adminsdk-flfvo-d5dce4bf44.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'techsavants-bdf20.appspot.com'
})

# Initialize Firebase Storage
bucket = storage.bucket()

# Initialize Firestore
db = firestore.client()

# Load the YOLOv8 model for object detection
object_detection_model = YOLO("Weights/lined_unlined.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for edge detection
running_avg_midpoint = None
alpha = 0.8  # Smoothing factor (close to 1 for more smoothing, close to 0 for less)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Object detection
        object_detection_results = object_detection_model(frame)

        # Get the confidence of 'unlined' class
        class_probs = object_detection_results[0].probs
        unlined_confidence = class_probs.data[1]  # Assuming you want the value at index 1
        print(unlined_confidence.item())

        if unlined_confidence.item() > 0.60:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            c_time = datetime.now().strftime("%H:%M:%S")
            c_date = datetime.now().strftime("%d-%m-%Y")

            url = "http://ipinfo.io/json"
            response = urlopen(url)
            data = json.load(response)
            lat, lon = map(float, data['loc'].split(','))

            # Create the directory if it doesn't exist
            os.makedirs("image/unlined_image", exist_ok=True)

            image_filename = f'image/unlined_image/{timestamp}.jpg'

            # Save the image locally
            cv2.imwrite(image_filename, frame)

            # Upload the image to Firebase Storage
            blob = bucket.blob(image_filename)
            blob.upload_from_filename(image_filename)

            # Get the URL of the uploaded image
            image_url = blob.public_url

            # Access Firestore and add the detected_objects data with the image URL
            detected_objects_collection = db.collection("unlined_objects")
            detected_objects_collection.add({
                'Date': c_date,
                'Time': c_time,
                'Issue': 'unlined',
                'Lattitude': lat,
                'Longitude': lon,
                'timestamp': timestamp,
                'image_url': image_url
            })

            # Edge detection
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and improve edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use Canny edge detection to detect edges in the frame
            edges = cv2.Canny(blurred, 50, 150)

            # Optional: Dilate the edges to enhance the edge features
            dilated_edges = cv2.dilate(edges, (1, 1), iterations=2)

            # Use Hough Line Transform to find lines in the dilated edges
            lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

            midpoints = []
            # If lines are found, store the midpoints of the lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = (y2 - y1) / (x2 - x1)  # Calculate the slope of the line
                    if 0.5 < abs(slope) < 2:  # Filter lines based on slope to isolate road lines
                        midpoints.append(((x1+x2)//2, (y1+y2)//2))  # Store the midpoints of the lines

            # Find the average midpoint to draw the path
            if midpoints:
                avg_midpoint = np.mean(midpoints, axis=0).astype(int)
                if running_avg_midpoint is None:
                    running_avg_midpoint = avg_midpoint
                else:
                    running_avg_midpoint = alpha * running_avg_midpoint + (1 - alpha) * avg_midpoint
                    running_avg_midpoint = running_avg_midpoint.astype(int)

                # Draw green line along the edge of the road
                cv2.line(frame, (running_avg_midpoint[0], frame.shape[0]), (running_avg_midpoint[0], 0), (0, 255, 0), 5)

            # Assume car position is at the bottom center of the frame
            car_position = (frame.shape[1]//2, frame.shape[0])
            # Set a threshold for how close to the edge the car needs to be before a warning is displayed
            warning_threshold = 30
            # Check if car is near the edge of the road and give a warning
            if abs(car_position[0] - running_avg_midpoint[0]) < warning_threshold:
                cv2.putText(frame, 'WARNING: Nearing Edge of Road', (450, 530), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Visualize the results on the frame
        annotated_frame = object_detection_results[0].plot()
        cv2.imshow("Object Detection and Edge Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
