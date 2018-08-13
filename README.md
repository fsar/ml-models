# ml-models
Just some machine learning models

* **cnn_binary_classifier.py**

Convolutional Neural Network in python for classifying two types of images. Using Keras and Tensorflow.<br /><br />
Structure:
- 2 convolution layers (32 feature detectors of size 3\*3)
- 2 max-pooling layers (pool size 2\*2)
- 1 input layer
- 1 hidden layer (128 nodes)
- 1 output layer

Used image augmentation ([Keras image preprocessing](https://keras.io/preprocessing/image/))
<br /> Trained on a dataset of 10,000 images. Achieved 85% on a training set and 82% on test set.
