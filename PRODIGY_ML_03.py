import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load images from the dataset directory
def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (100, 100))  # Resize the image
            images.append(img.flatten())  # Flatten the image into a 1D array
            labels.append(label)
    return images, labels

# Load images of cats and dogs
cat_images, cat_labels = load_images("cats_dataset", 0)  # Assuming "cats_dataset" contains images of cats
dog_images, dog_labels = load_images("dogs_dataset", 1)  # Assuming "dogs_dataset" contains images of dogs

# Concatenate cat and dog images and labels
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM model
svm_model = SVC(kernel='linear')

# Training the SVM model
svm_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
