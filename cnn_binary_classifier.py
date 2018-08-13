# Building a Convolutional Neural Network to classify images of cats and dogs

# Importing the tensorflow (and keras) libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# PART 1: Building the CNN

# Initialising the CNN as a sequence of layers
# Creating an object of the Sequential class
classifier = Sequential()

# Convolution layer (.add method to add a layer to the network)
# Convolution2D: 32 = n° of filters. 3, 3 size of the filter/feature detector.
# Input shape = size of the images to classify. relu as activation function.
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling - Downsampling the feature maps with a 2x2 pool 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer + pooling
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening - Transforming all our pools into one single vector.
# Preparation for the ANN
classifier.add(Flatten())

# Full connection
# 1 hidden layer with 128 neuros
# Output layer: sigmoid as the outcome is binary (cat or dog)
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# More info on optimizers: https://keras.io/optimizers/
# More info on loss functions: https://keras.io/losses/
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# PART 2: Fitting the CNN to the images
# More info: https://keras.io/preprocessing/image/

# Import the libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation - Applying different transformations to the images
# Goal is to reduce over fitting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

# Results: 
# Accuracy 85% on training set. 82% on test set.