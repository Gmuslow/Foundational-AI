import matplotlib
matplotlib.use('TkAgg')
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import mlp
from sklearn.model_selection import train_test_split

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (np.array(x_train), np.array(y_train)),(np.array(x_test), np.array(y_test))

#
# Set file paths based on added MNIST Datasets
#
input_path = 'data'
training_images_filepath = "data\\train-images.idx3-ubyte"
training_labels_filepath = "data\\train-labels.idx1-ubyte"
test_images_filepath = "data\\t10k-images.idx3-ubyte"
test_labels_filepath = "data\\t10k-labels.idx1-ubyte"

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(y_train)
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# One-hot encode the output labels
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

print(f"x_train_flattened shape: {x_train_flattened.shape}")
print(f"y_train_one_hot shape: {y_train_one_hot.shape}")

# Split the training data into training and validation sets

x_train_flattened, x_val_flattened, y_train_one_hot, y_val_one_hot = train_test_split(
    x_train_flattened, y_train_one_hot, test_size=0.2, random_state=42
)

print(f"x_train_flattened shape: {x_train_flattened.shape}")
print(f"x_val_flattened shape: {x_val_flattened.shape}")
print(f"y_train_one_hot shape: {y_train_one_hot.shape}")
print(f"y_val_one_hot shape: {y_val_one_hot.shape}")

# Initialize layers with consistent dimensions
layers = (
    mlp.Layer(fan_in=784, fan_out=784, activation_function=mlp.Linear()),
    mlp.Layer(fan_in=784, fan_out=64, activation_function=mlp.Relu()),  # First hidden layer
    mlp.Layer(fan_in=64, fan_out=32, activation_function=mlp.Relu()),  # Second hidden layer
    mlp.Layer(fan_in=32, fan_out=10, activation_function=mlp.Softmax()),  # Output layer
)

multi_layer_perceptron = mlp.MultilayerPerceptron(layers,
    verbose=False)

loss_function = mlp.CrossEntropy()
multi_layer_perceptron.train(
    x_train_flattened, 
    y_train_one_hot, 
    x_val_flattened, 
    y_val_one_hot, 
    loss_function, 
    learning_rate=0.01,
    batch_size=32,
    epochs=50
)

num_right = 0
# Evaluate the multi_layer_perceptron using the testing set
test_output = multi_layer_perceptron.forward(x_test_flattened)
predicted_labels = np.argmax(test_output, axis=1)
true_labels = y_test

# Calculate accuracy
num_right = np.sum(predicted_labels == true_labels)
accuracy = num_right / len(true_labels)

print(f"Number of correct predictions: {num_right}")
print(f"Total number of predictions: {len(true_labels)}")
print(f"Accuracy: {accuracy * 100:.2f}%")