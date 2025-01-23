import cv2

# Load the Haar Cascade classifier

model = 'haarcascade_frontalface_alt.xml'

cascade = cv2.CascadeClassifier(model)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the Raspberry Pi camera

# Set the camera resolution (optional for performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
process_interval = 5  # Process every 5th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_interval != 0:
        continue

    # Convert frame to grayscale for the classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans in the frame
    humans = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the detected humans
    for (x, y, w, h) in humans:
        # Draw bounding box around the detected person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
