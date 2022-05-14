# Python-Programming--Final-Project
Final Project- AI 551
Purpose:
This project aims to classify images as a properly fitted mask (Fully covered: mask covers nose and mouth), not properly fitted (partially covered: the nose or the mouth), and not wearing a mask (not covered). 

Dataset:
The dataset was obtained from Kaggle ( https://www.kaggle.com/datasets/jamesnogra/face-mask-usage ). The dataset partitioned into 70 % training set, 20 % validation set, 10 % testing set. The original dataset involves 4 classes: fully covered, not covered, not face, and partially covered. In this project, I used 3 classes, fully covered, not covered, and partially covered to train, validate, and test my CNN. I removed the not face class from the dataset. This dataset is imbalanced; the partially covered class has few images compared to the other two classes.  
I used Open-Source Computer Vision Library (OpenCV) to handle my image dataset.

Model:
I built Convolutional Neural Network (CNN) to classify given images into fully covered, not covered, and partially covered.  My CNN consists of a stack of 2D convolutional layers with relu activation function, and max-pooling layers interchangeably. 

Results:
The CNN model achieved an accuracy of 97.83 % on the test set and a loss of 0.1499. I plotted the confusion metric to visualize model performance on the test set. We can infer that this model can predict the not covered class with high performance compared to the other two classes. And the model can predict a fully covered class with relatively higher accuracy than it can predict a partially covered. To further investigate the model, we draw predictions on a single image from each class. We found that this model can classify an image of a face from the fully covered class correctly with a likelihood of 99.99%. And this model can classify an image of a face from the not covered class correctly with a likelihood of 99.99 %. But the model failed to classify partially covered classes. Model misclassified an image from the partially covered class as not covered with a likelihood of 99.32 %.

