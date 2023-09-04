# Face Mask Usage Classifier

**Project Status:** Completed

## Project Overview
The primary objective of this project is to classify images into three categories: 
1. Fully covered face masks (where the mask covers both the nose and mouth).
2. Partially covered face masks (where either the nose or mouth is not fully covered).
3. No face mask worn (not covered).

This project was carried out individually as a final Python programming assignment.

### Methods Used
- Machine Learning
- Deep Learning
- Predictive Modeling

### Technologies
- Jupyter Notebook

### Libraries Used
- OpenCV
- TensorFlow
- NumPy
- Matplotlib

## Project Description

### Data Source
The dataset used for this project was obtained from Kaggle ([Face Mask Usage Dataset](https://www.kaggle.com/datasets/jamesnogra/face-mask-usage)). 

### Data Partitioning
The dataset was divided into three subsets:
- 70% for the training set
- 20% for the validation set
- 10% for the testing set

The code for data partitioning can be found in a separate IPython Notebook.

### Data Preprocessing
- The images in all three sets (training, validation, and testing) were converted into arrays using OpenCV's `imread` function.
- Labels and features were created, with labels representing the target variables (fully covered, not covered, and partially covered) and features representing the image data.
- Feature normalization was performed by dividing the pixel values by 255.
- Labels were converted to a binary class matrix.

### Convolutional Neural Network (CNN)
A Convolutional Neural Network (CNN) was designed to classify the images into the three categories. The CNN architecture consists of:
- Two-dimensional convolutional layers with ReLU activation functions.
- Max-pooling layers placed alternately with convolutional layers.
- Filters with sizes of 32 for the first two convolutional layers and 64 for the last two. Various filter sizes, including 32, 64, and 128, were experimented with.
- A window size of (3,3) and "same" padding.
- A dropout of 0.25 was added after the max-pooling layers.
- The final layers include a flatten layer, two dense layers (with 512 neurons and ReLU activation), a dropout layer (0.50), and a dense layer with 3 neurons (softmax activation) for multiclass classification.
- The CNN model was compiled with categorical cross-entropy as the loss function, Adam optimizer, and accuracy as the performance metric.
- Early stopping was implemented to prevent overfitting. The training process stops if the validation loss does not improve after 5 epochs.

### Model Performance
The CNN model achieved an accuracy of 97.83% on the test set with a loss of 0.1499. A confusion matrix was created to visualize the model's performance. The model demonstrates strong performance in predicting the "not covered" class, surpassing the other two classes. Additionally, it exhibits relatively higher accuracy in predicting the "fully covered" class compared to the "partially covered" class.

### Individual Image Predictions
For further examination, predictions were made on single images from each class. The model correctly classified an image of a fully covered face with a likelihood of 99.99% and an image of a not covered face with a likelihood of 99.99%. However, the model faced challenges in classifying the "partially covered" class, misclassifying an image as "not covered" with a likelihood of 99.32%.


## Bugs/Notes
To resolve the dead kernel issue during model fitting, please use the following code:

os.environ['KMP_DUPLICATE_LIB_OK']='True'

I encountered persistent error messages indicating that the kernel had died and required restarting. The provided code resolved the issue for me. Credit to (https://www.kaggle.com/product-feedback/41221).

## Project Requirements
- Data exploration/descriptive statistics
- Data processing/cleaning



