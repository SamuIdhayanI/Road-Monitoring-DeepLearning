import cv2
import torch
import threading
import queue
from ultralytics import YOLO

# Load your Pytorch models
model1 = YOLO("Weights/all_detec.pt")
model2 = YOLO("Weights/lined_unlined.pt")


def process_model1(frame_queue, output_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        with torch.no_grad():
            output = model1(frame, conf=0.6)
        output_queue.put(output)


def process_model2(frame_queue, output_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        with torch.no_grad():
            output = model2(frame)
        output_queue.put(output)


def main():
    frame_queue1 = queue.Queue()
    frame_queue2 = queue.Queue()
    output_queue1 = queue.Queue()
    output_queue2 = queue.Queue()

    thread1 = threading.Thread(target=process_model1, args=(frame_queue1, output_queue1))
    thread2 = threading.Thread(target=process_model2, args=(frame_queue2, output_queue2))

    thread1.start()
    thread2.start()

    cap = cv2.VideoCapture(0)

    # Define the desired resolution (width x height)
    desired_resolution = (640, 640)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the desired resolution
        resized_frame = cv2.resize(frame, desired_resolution)

        frame_queue1.put(resized_frame.copy())
        frame_queue2.put(resized_frame.copy())

        # Get the output from both models
        output1 = output_queue1.get()
        output2 = output_queue2.get()

        # Copy the resized frame to overlay annotations from both models
        annotated_frame = resized_frame.copy()

        # Overlay the annotations from model1
        annotated_frame1 = output1[0].plot(img=annotated_frame)

        # Overlay the annotations from model2
        annotated_frame2 = output2[0].plot(img=annotated_frame1)

        # Display the combined annotated frame
        cv2.imshow('Output', annotated_frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_queue1.put(None)
    frame_queue2.put(None)
    thread1.join()
    thread2.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()