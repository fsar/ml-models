# ml-models
*Just some machine learning models coded during my free time...*


* **ae_model_movie_rating.py** - KAGGLE competition dataset ([Predict movie ratings](https://www.kaggle.com/c/predict-movie-ratings)) - **RMSE scored = 0.95153** <br />

Stacked Auto-Encoder in python for predicting movie ratings. Using [PyTorch](https://pytorch.org/).<br />
Structure:
- Encoder (1 input layer + 1 layer of 20 nodes)
- Code (10 nodes)
- Decoder (1 layer of 20 nodes + 1 output layer)

Pre-processed the data.
<br /><br />


* **cnn_binary_classifier.py**

Convolutional Neural Network in python for classifying two classes of images. Using [Keras](https://keras.io/) and Tensorflow.<br />
Structure:
- 2 convolution layers (32 feature detectors of size 3\*3)
- 2 max-pooling layers (pool size 2\*2)
- 1 input layer
- 2 hidden layers (64 nodes each)
- 1 output layer

Used image augmentation ([Keras image preprocessing](https://keras.io/preprocessing/image/))
<br /> Trained on a dataset of 10,000 images. Achieved 85% on a training set and 82% on test set.
<br /><br />


* **ann_binary_classifier.py**<br />

Artificial Neural Network in python for churn modelling. Using [Scikit-learn](http://scikit-learn.org/), [Keras](https://keras.io/) and Tensorflow.<br />
Structure:
- 1 input layer
- 2 hidden layers (16 nodes each) with dropout
- 1 output layer

Pre-processed the data. Applied K-fold Cross Validation and Grid Search to optimize the parameters. 
<br /><br />


* **rnn_regressor_stock_price.py**<br />

Recurrent Neural Network **LSTM** in python for predicting stock price. Using [Scikit-learn](http://scikit-learn.org/), [Keras](https://keras.io/) and Tensorflow.<br />
Structure:
- 4 LSTM layers (50 nodes each)
- 1 output layer

Pre-processed the data.
<br /><br />
