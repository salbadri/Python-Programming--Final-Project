# Face Mask Usage

#### Project Status: [Completed]

## Project Introduction/Objective
The goal of this project is to classify images based on face mask usage into three categories: properly fitted mask (fully covered: mask covers nose and mouth), not properly fitted (partially covered: either the nose or the mouth is not covered), and not wearing a mask (not covered). This is a solo final project. 

### Methods Used
* Machine Learning
* Deep Learning
* Predictive Modeling

### Technologies
* Jupyter

### Used Libraries
* OpenCV
* TensorFlow
* NumPy
* Matplotlib

## Project Description

The dataset was obtained from Kaggle ( https://www.kaggle.com/datasets/jamesnogra/face-mask-usage ). I divided the dataset into three parts: 70% for the training set, 20% for the validation set, and 10% for the testing set. The code for data partitioning is provided in a separate IPython Notebook (ipynb file).

The original dataset had four classes: fully covered, not covered, not face, and partially covered. For this project, I utilized three classes: fully covered, not covered, and partially covered to train, validate, and test my Convolutional Neural Network (CNN), removing the not face class from the dataset.

For preprocessing, I converted the images in the training set, validation set, and testing set into arrays using the imread function from OpenCV. I created labels and features, with labels representing the target variables (y) for fully covered, not covered, and partially covered, and features (x) extracted from the images.

I normalized the features in all three sets (training, validation, and testing) by dividing them by 255. The labels in all three sets were converted to a binary class matrix.

I constructed a Convolutional Neural Network (CNN) to classify images into fully covered, not covered, and partially covered categories. My CNN comprises a stack of 2D convolutional layers with ReLU activation functions and max-pooling layers, alternating between them.

Filter sizes were set to 32 for the first two convolutional layers and 64 for the last two convolutional layers. I experimented with different filter sizes, including 32, 64, and 128. The window size used was (3,3), with padding set to "same." The activation function used was ReLU.

Two max-pooling layers with a pool size of (2, 2) were added, one after the second convolutional layer and another after the fourth convolutional layer.

Following the max-pooling layers, a dropout of 0.25 was incorporated.

The last three layers of the CNN consist of a flatten layer and two dense layers. The flatten layer serves as a transition from a 2-dimensional array to a dense layer. The dense layer comprises 512 neurons with a ReLU activation function and a dropout of 0.50. The final layer is a dense layer with 3 neurons (reflecting the 3 classes) and a softmax activation function, which is suitable for multiclass classification.

I compiled the neural network using categorical cross-entropy as the loss function due to the multiclass classification nature of the problem. The optimizer used was Adam, and accuracy was chosen as the performance metric.

To prevent overfitting, I implemented early stopping. The training process monitors the validation loss, and if it does not improve after 5 epochs, training is halted.

The CNN model achieved an accuracy of 97.83% on the test set with a loss of 0.1499. I also generated a confusion matrix to visually assess the model's performance on the test set. It is evident that the model excels in predicting the not covered class, outperforming the other two classes. Additionally, the model exhibits relatively higher accuracy in predicting the fully covered class compared to the partially covered class. For further investigation, I conducted predictions on single images from each class. The model correctly classified an image of a fully covered face with a likelihood of 99.99% and an image of a not covered face with a likelihood of 99.99%. However, the model struggled with the partially covered class, misclassifying an image as not covered with a likelihood of 99.32%.


## Bugs/Notes
To resolve the dead kernel issue during model fitting, please use the following code:

os.environ['KMP_DUPLICATE_LIB_OK']='True'

I encountered persistent error messages indicating that the kernel had died and required restarting. The provided code resolved the issue for me. Credit to (https://www.kaggle.com/product-feedback/41221).

## Project Requirements
- Data exploration/descriptive statistics
- Data processing/cleaning



