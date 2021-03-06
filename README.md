# Face Mask Usage

#### -- Project Status: [Completed]

## Project Intro/Objective
The goal of this project is to classify images as a properly fitted mask (Fully covered: mask covers nose and mouth), not properly fitted (partially covered: the nose or the mouth), and not wearing a mask (not covered).
This is a solo final project (Python Programming)

### Methods Used
* Machine Learning
* Deep Learning
* Predictive Modeling


### Technologies
* Jupyter 

### used Libraries
* OpenCV
*  TensorFlow
* Numpy
* Matplotlib

## Project Description

The dataset was obtained from Kaggle ( https://www.kaggle.com/datasets/jamesnogra/face-mask-usage ). I partitioned the dataset into 70 % training set, 20 % validation set, 10 % testing set. I attached code for data partitioning in a separate ipynb file.

The original dataset has 4 classes: fully covered, not covered, not face, and partially covered. In this project, I used 3 classes, fully covered, not covered, and partially covered to train, validate, and test my CNN. I removed the not face class from the dataset.
Preprocessing my training set, validation set, and testing set: converted data into an array, using imread from OpenCV.
I created my labels and features. Labels are the target variables (y) that are fully covered, not covered, and partially covered. Features (x) that are extracted from the images.

I Normalized features, in the three sets (training, validation, and testing) by dividing by 255. Labels, for the three sets (training, validation, and testing) converted to a binary class matrix.


I built a Convolutional Neural Network (CNN) to classify given images into fully covered, not covered, and partially covered. 
My CNN consists of a stack of 2D convolutional layers with relu activation function, and max-pooling layers interchangeably.

Filters: size 32 for the first two convolutional layers, and filters of 64 for the last two convolutional layers. I tried different filter sizes 32, 64, and 128. 

Window size: (3,3). 

Padding is the same. 

The activation function: relu activation function. 

Two maxpooling layers were used with a pool size of (2, 2); one layer after the second convolution layer, and another after the fourth convolutional layer.
 
 After the maxpooling layers, a dropout of 0.25 was added.

The last 3 layers of the CNN are a flatten layer and 2 dense layers. The flatten layer is a transitioning layer from a 2-dimensional array to a dense layer. 
Dense layer of 512 neurons and activation function: relu. 

Dropout of 0.50. The model ends with a Dense layer with 3 neurons (because we have 3 classes), and softmax activation function; softmax as it???s a multiclass classification.

I compiled the neural network. The loss function is categorical cross-entropy because it is a multiclass classification. The optimizer is Adam, and accuracy is the performance metric.

I implemented early stopping to prevent my model from overfitting. we monitor validation loss if it is not improving after 5 epochs, stop the training data.


The CNN model achieved an accuracy of 97.83 % on the test set and a loss of 0.1499. I plotted the confusion metric to visualize model performance on the test set. We can infer that this model can predict the not covered class with high performance compared to the other two classes. And the model can predict a fully covered class with relatively higher accuracy than it can predict a partially covered. To further investigate the model, we draw predictions on a single image from each class. We found that this model can classify an image of a face from the fully covered class correctly with a likelihood of 99.99%. And this model can classify an image of a face from the not covered class correctly with a likelihood of 99.99 %. But the model failed to classify partially covered classes. Model misclassified an image from the partially covered class as not covered with a likelihood of 99.32 %.

## bugs/notes
To fix dead kernel issue when fitting the model use the following code:

os.environ['KMP_DUPLICATE_LIB_OK']='True'

I kept getting error messages informing me that the kernel have died and restart needed. The above code solved the problem for me. credit to (https://www.kaggle.com/product-feedback/41221) 

## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning



