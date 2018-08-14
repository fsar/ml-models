# Building a Convolutional Neural Network to classify images of cats and dogs

# Importing the tensorflow (and keras) libraries
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# PART 1: Building the CNN

def build_model(dropout_rate = 0.5, input_shape = (64, 64, 3)):
    
    # Initialising the CNN as a sequence of layers
    # Creating an object of the Sequential class
    model = Sequential()
    
    # Convolution layer (.add method to add a layer to the network)
    # Convolution2D: 32 = n° of filters. 3, 3 size of the filter/feature detector.
    # Input shape = size of the images to classify. relu as activation function.
    model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    
    # Pooling - Downsampling the feature maps with a 2x2 pool 
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer + pooling
    model.add(Convolution2D(32, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Flattening - Transforming all our pools into one single vector.
    # Preparation for the ANN
    model.add(Flatten())
    
    # Full connection
    # 2 hidden layer with 64 nodes each
    # Output layer: sigmoid as the outcome is binary (cat or dog)
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(dropout_rate/2))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    # More info on optimizers: https://keras.io/optimizers/
    # More info on loss functions: https://keras.io/losses/
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


# PART 2: Fitting the CNN to the images
# More info: https://keras.io/preprocessing/image/

def run_training(model, batch_size = 32, epochs = 10):
    
    # Image augmentation - Applying different transformations to the images
    # Goal is to reduce over fitting
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
                                        
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = batch_size,
                                                class_mode = 'binary')
    model.fit_generator(training_set,
                             steps_per_epoch = 8000/batch_size,
                             epochs = epochs,
                             validation_data = test_set,
                             validation_steps = 2000/batch_size)
    
    model.save('cnn_binary_classifier_trained.h5')
    
    return model, training_set.class_indices


# PREDICTION for 1 image

def check(classifier_model, class_indices, path):
    
    # Load the image using image class from keras
    # Size 64*64 as per the model was trained
    test_image = image.load_img(path, target_size = (64, 64))                                         
    test_image = image.img_to_array(test_image)
    
    # Add a dimension to the image  
    test_image = np.expand_dims(test_image, axis = 0)
    
    # Run the prediction
    result = classifier_model.predict(test_image)
    if result[0][0] == 1:
        prediction = [key for key, value in class_indices.items() if value == 1][0]
    else:
        prediction = [key for key, value in class_indices.items() if value == 0][0]  

    return (path, prediction)                                         

"""
# PREDICTION for 1 or more images
from tensorflow.keras.preprocessing import image

pred_datagen = ImageDataGenerator(rescale = 1./255)
pred_set = pred_datagen.flow_from_directory('dataset/prediction_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = None,
                                            shuffle = False)

filenames = pred_set.filenames
nb_samples = len(filenames)

predict = classifier.predict_generator(pred_set, steps = nb_samples)
label_dict = training_set.class_indices
"""


def main():
    
    # Build the model
    classifier = build_model(dropout_rate = 0.2, input_shape = (128, 128, 3))
    
    # Train the model
    classifier_trained, d_class_indices = run_training(classifier, batch_size = 32, epochs = 100)
    
    # Get the prediction for 1 image
    prediction_path = 'dataset/prediction_set/cat_or_dog.jpg'
    prediction_result = check(classifier_trained, d_class_indices, prediction_path)
    print (prediction_result)

    
""" Main """    
if __name__ == "__main__":
    main()
