import cv2
import ollama
import time
import numpy as np
from pprint import pprint

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the webcam feed
    cv2.imshow("Webcam Feed", frame)

    # Allow the user to press 'q' to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Start time for LLaVA model request
    start_time = time.time()

    # Convert the frame to PNG format in memory (not saving to disk)
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Perform LLaVA model analysis using the frame
    res = ollama.chat(
        model='llava',
        messages=[
            {'role': 'user',
             'content': 'Analyze this picture: size and calories, and provide exact numbers.',
             'images': [img_bytes]
             }
        ]
    )

    # End time for LLaVA model request
    end_time = time.time()

    # Print the analysis result
    pprint(res['message']['content'])

    # Print time taken for LLaVA analysis
    print("Exact time:", end_time - start_time, "seconds")

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
