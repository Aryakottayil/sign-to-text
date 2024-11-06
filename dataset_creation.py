import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe Hand class.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Path to your dataset directory
dataset_dir = "American_Sign_Language_Letters_Multiclass"
output_csv = "hand_landmarks.csv"

# Open CSV file to write landmark data.
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header: 42 features (21 points x 2 coordinates) + label
    header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
    writer.writerow(header)

    # Iterate over each subfolder (representing each ASL letter/symbol)
    for label_folder in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        # Process each image in the subfolder
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue

            # Convert the image to RGB format (required for MediaPipe)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            # Extract and save landmarks if detected
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmark_list = []
                    for landmark in hand_landmarks.landmark:
                        # Normalize coordinates
                        landmark_list.extend(
                            [landmark.x, landmark.y]
                        )  # Normalized values

                    # Write the landmark data to the CSV with the label as the last column
                    writer.writerow(landmark_list + [label_folder])

# Release MediaPipe resources.
hands.close()
cv2.destroyAllWindows()
