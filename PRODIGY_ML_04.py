import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load hand gesture images and corresponding labels
def load_data(data_dir):
    images = []
    labels = []
    gesture_names = os.listdir(data_dir)
    for label, gesture_name in enumerate(gesture_names):
        gesture_dir = os.path.join(data_dir, gesture_name)
        for filename in os.listdir(gesture_dir):
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(gesture_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (100, 100))  # Resize the image to a fixed size
                images.append(img.flatten())  # Flatten the image into a 1D array
                labels.append(label)
    return np.array(images), np.array(labels)

# Load hand gesture images and labels
X, y = load_data("hand_gestures_dataset")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
