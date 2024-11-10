import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# Load the TensorFlow model
print("Loading TensorFlow model...")
model = tf.keras.models.load_model("asl_model.h5")
print("Model loaded successfully.")

# Initialize MediaPipe Hands
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)
mp_draw = mp.solutions.drawing_utils
print("MediaPipe Hands initialized.")

# Label encoder (used for mapping output index to labels)
label_encoder = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]  # Replace with your actual labels

# Start webcam feed
print("Starting webcam feed...")
cap = cv2.VideoCapture(0)

# Sliding window for smoothing predictions
window_size = 5
predictions = deque(maxlen=window_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract normalized landmark coordinates
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.extend([landmark.x, landmark.y])

            # Convert landmarks to a numpy array and reshape to match model input
            input_data = np.array([landmark_list], dtype=np.float32).reshape(
                1, 21, 2, 1
            )

            # Run inference
            output_data = model.predict(input_data)
            predicted_index = np.argmax(output_data)
            confidence = np.max(output_data)

            # Check if the confidence is above the threshold
            if confidence > 0.5:
                # Map the index to the label
                predicted_label = label_encoder[predicted_index]

                # Add prediction to the sliding window
                predictions.append(predicted_label)

                # Get the most common prediction in the sliding window
                most_common_prediction = max(set(predictions), key=predictions.count)

                # Calculate the bounding box around the hand
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                # Convert normalized coordinates to pixel values
                h, w, _ = frame.shape
                x_min = int(x_min * w)
                y_min = int(y_min * h)
                x_max = int(x_max * w)
                y_max = int(y_max * h)

                # Display the predicted letter near the bounding box
                cv2.putText(
                    frame,
                    most_common_prediction,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

    # Display the frame
    cv2.imshow("ASL Sign Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Resources released.")
