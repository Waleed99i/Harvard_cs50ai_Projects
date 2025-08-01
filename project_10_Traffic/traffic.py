import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network model
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate model performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Iterate over each traffic sign category (0 to NUM_CATEGORIES - 1)
    for class_label in range(NUM_CATEGORIES):
        class_folder_path = os.path.join(data_dir, str(class_label))

        # Iterate over each image file in the category folder
        for filename in os.listdir(class_folder_path):
            image_full_path = os.path.join(class_folder_path, filename)

            # Read the image using OpenCV
            image_array = cv2.imread(image_full_path)

            # Resize image to match the model's input dimensions
            image_array = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))

            # Store image data and its label
            images.append(image_array)
            labels.append(class_label)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    input to the neural network is typically of a shape (IMG_WIDTH, IMG_HEIGHT, 3),
    representing an image with width IMG_WIDTH, height IMG_HEIGHT, and 3 color channels (RGB).
    The output layer should have NUM_CATEGORIES units, one for each traffic sign category.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer. Pool using a 2x2 filter
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Another convolutional layer. Learn 64 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),

        # Another max-pooling layer. Pool using a 2x2 filter
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Hidden layer with 128 units
        tf.keras.layers.Dense(128, activation="relu"),

        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer with NUM_CATEGORIES units (one for each traffic sign category)
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

