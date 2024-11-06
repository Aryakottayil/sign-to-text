import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten,
    LSTM,
    TimeDistributed,
    Reshape,
)
import numpy as np

# Load the CSV data
data = pd.read_csv("hand_landmarks.csv")

# Separate features and labels
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values  # The last column (label)

# Encode the labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Print the label names
print("Label names:", label_encoder.classes_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Print the shape of X_train before reshaping
print("Shape of X_train before reshaping:", X_train.shape)

# Reshape the data to fit the Conv2D layer input requirements
image_size = 21  # Number of landmarks
img_channel = 2  # x and y coordinates

# Check if the size matches
if X_train.shape[1] != image_size * img_channel:
    raise ValueError(
        f"Cannot reshape array of size {X_train.shape[1]} into shape ({image_size}, {img_channel})"
    )

X_train = X_train.reshape(-1, image_size, img_channel, 1)
X_test = X_test.reshape(-1, image_size, img_channel, 1)

# Build the new model architecture
model = Sequential()
# input layer
# Block 1
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(image_size, img_channel, 1),
    )
)
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.4))

# fully connected layer
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

# output layer
model.add(Dense(y_train.shape[1], activation="softmax"))

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Save the model in the standard TensorFlow format
model.save("asl_model.h5")

print("Model has been saved as 'asl_model'")
