import cv2
import numpy as np
from time import sleep

# Load the pre-trained Haar Cascade for detecting people

model = 'haarcascade_frontalface_alt.xml'

hog_cascade = cv2.CascadeClassifier(model)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the Raspberry Pi camera

# Set the camera resolution (optional for performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


# Let autofocus complete before starting frame capture
sleep(2)  # Adjust the sleep time based on autofocus speed

try:
    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale (Haar Cascades work on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people in the image
        people = hog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected people
        for (x, y, w, h) in people:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with detection
        cv2.imshow("Human Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cv2.destroyAllWindows()
