#Step1:importing the dependencies
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
#Step2: Load the MNIST dataset
mnist=fetch_openml("mnist_784")
#Step3: Split the data and labels
X=mnist.data.astype(float)
y=mnist.target.astype(int)
#Step4: Normalize the data (scaling to [0, 1])
X=X/255
#Step5: Split the dataset into training and test sets
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
#Step6: Optionally scale the data (standardization to mean=0 and variance=1)
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#Step7: Define the SVM model
model=SVC(kernel='rbf',gamma='scale')
#Step8: Train the model
model.fit(X_train,y_train)
#Step9: Predict on the test data
y_pred=model.predict(X_test)
#Step10: Calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to display images along with predictions
def plot_images(images, labels, predictions=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        if predictions is None:
            plt.title(f'Label: {labels[i]}')
        else:
            plt.title(f'True: {labels[i]}\nPred: {predictions[i]}')
        plt.axis('off')
    plt.show()

# Display first 10 test images, their true labels, and the predicted labels
plot_images(X_test[:10], y_test[:10], y_pred[:10])