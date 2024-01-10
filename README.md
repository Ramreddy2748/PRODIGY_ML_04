## **Hand Gesture Recognition using Convolutional Neural Network (CNN)**

**Overview**

This project focuses on building a Convolutional Neural Network (CNN) for hand gesture recognition. The dataset used consists of infrared images of hand gestures captured by a Leap Motion sensor. The CNN is designed to classify these gestures into distinct categories.

**Prerequisites**

Make sure you have the following dependencies installed:

NumPy
os
PIL (Python Imaging Library)
Matplotlib
Keras
scikit-learn
You can install these dependencies using the following command:

pip install numpy pillow matplotlib keras scikit-learn

**Data Processing**

The dataset is organized into folders, each representing a different subject and containing subfolders for each gesture. The code in Data processing reads and processes the images, creating numerical classifiers and reshaping the data for model training.

**Converting Data to Categorical Values**

The script converts the numerical classifiers to categorical values using one-hot encoding. This step is crucial for training a multiclass classification model.

**Training and Validation Dataset**

The dataset is split into training, validation, and test sets using the train_test_split function from scikit-learn. This ensures that the model is trained on a diverse set of data and evaluated on separate unseen data.

**Building the CNN Model**

The architecture of the CNN model is defined using Keras. It consists of multiple convolutional layers followed by max-pooling layers and fully connected layers. The final layer uses softmax activation for multiclass classification.

**Training the Model**

The model is compiled with the RMSprop optimizer and categorical crossentropy loss. It is then trained on the training set, and the validation set is used to monitor the model's performance and prevent overfitting.

**Model Evaluation**

The README includes visualizations of training and validation accuracy and loss over epochs. The trained model achieves impressive accuracy on the validation and test sets.

Usage

Run the data processing script.
Execute the script for converting data to categorical values.
Train the model using the provided code.
Evaluate the model on the test set.
Results

The model achieves an **accuracy of 99.95%** on the test set, demonstrating its effectiveness in hand gesture recognition.

