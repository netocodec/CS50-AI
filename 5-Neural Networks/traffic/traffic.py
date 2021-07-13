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

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
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
    images_result = []
    labels_result = []

    for dir_name in os.listdir(data_dir):
        path = os.path.join(data_dir, dir_name)
        directory_id = int(dir_name)
        images_list = os.listdir(path)

        print(f"Loading {path} folder...")
        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)
            image_cv = cv2.imread(image_path)
            image_resized = cv2.resize(image_cv, (IMG_WIDTH, IMG_HEIGHT))

            images_result.append(image_resized)
            labels_result.append(directory_id)

    return (images_result, labels_result)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Filters parameters
    filters_number = 32
    conv_kernel_size = (2, 2)
    max_pooling = (2,2)
    dense_layers = 16
    activation_type = 'relu'
    dropout_rate = 0.2

    model = tf.keras.Sequential([
        #Convolutional Layer in a 2D context
        tf.keras.layers.Conv2D(filters_number, conv_kernel_size, activation=activation_type, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Adding Max Pooling Layer
        # This downsamples the input representation by taking the maximum value.
        tf.keras.layers.MaxPooling2D(pool_size=max_pooling),

        # Adjust the regular densely-connected NN Layer
        tf.keras.layers.Dense(dense_layers, input_shape=(NUM_CATEGORIES,), activation=activation_type),

        # Flattens the input
        tf.keras.layers.Flatten(),

        # Adding Dropout layer to adjust the frequency of rate
        tf.keras.layers.Dropout(dropout_rate),

        # Dense layer with Num_Categories for the output
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])


    # Compile Our Model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()

