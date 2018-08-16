# Recurrent Neural Network

# Import the libraries
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


def preprocess_data(training_set_scaled, sc):
    # Create a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    
    # Starting at 60 since for each i we need the 60 previous timesteps
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape (to the input shape that is expected by the LSTM layer)
    # batch_size = total number of observations we have (number of stock prices from 2012 to 2016) 
    # n° of timesteps = 60 (we choose 60 days)
    # input_dim = indicators/predictors (here only the open price of google, so 1)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train
    
def build_model(X_train, dropout_rate = 0.2, optimizer = 'adam'):
    
    # Initialise the RNN
    model = Sequential()
    
    # First layer of LSTM + dropout
    # input_shape: the first dimension of X_train is taken into account autmatically. Only need to specify the two last dims.
    model.add(LSTM(units = 50, return_sequences = True,  input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    
    # Second layer of LSTM + dropout
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(dropout_rate))
    
    # Third layer of LSTM + dropout
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(dropout_rate))
    
    # Fourth layer of LSTM + dropout
    model.add(LSTM(units = 50))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units = 1))

    # Compile
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    return model


def run_training(model, X_train, y_train, epochs = 100, batch_size = 32):
    
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

    # Save the model in H5 file
    model.save('rnn_regressor_stock_price.h5')
    

# Predict the stock price 
def check(model, dataset_train, dataset_test, sc):
    
    """
    We trained the model to be able to predict the stock price at t+1 based on
    the 60 previous stock prices. Therefore, to predict the stock prices in
    January 2017, we will need the 60 previous stock prices before the actual 
    day in January (we will need the training and test set).
    Start with concatenation.
    """
    
    # Create the test set
    # Concatenate both the sets (only 'open' columns).
    # Not using 'real_stock_price' because we will have to apply sc on the concatenation 
    # axis = 0 to concatenate the lines (not the columns)
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    
    # We want the 60 timesteps before the 1st financial day of January (3rd)
    # First substraction gives 3rd of January. Whole substraction gives the lower bound.
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    
    # We specify we want 1 column (1) and -1 means numpy will figure out the number of rows
    inputs = inputs.reshape(-1, 1)
    
    # Scale the inputs. sc is already fitted.
    inputs = sc.transform(inputs)

    # Create X test
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    
    # Reshape X_test
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predict stock price
    predicted_stock_price = model.predict(X_test)
    
    # Unscale the result
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price
    
    
def main():
    
    # Import the datasets (train + test)
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    # Feature scaling - Create object for normalisation
    # We create the training set and then apply normalisation on the data
    sc = MinMaxScaler(feature_range = (0 , 1))
    training_set = dataset_train.iloc[:, 1:2].values
    training_set_scaled = sc.fit_transform(training_set)
    
    # Check if the model has already been trained
    model_file = 'rnn_regressor_stock_price.h5'
    if os.path.exists(model_file):
        # Load the model
        print ('Loading the model from {0}/{1}'.format(os.getcwd(), model_file))
        regressor = load_model(model_file)
    else :        
        # Create model and train it
        print ('Training the model')
        X_train, y_train = preprocess_data(training_set_scaled, sc)
        regressor = build_model(X_train, dropout_rate = 0.2, optimizer = 'adam')
        run_training(regressor, X_train, y_train, epochs = 100, batch_size = 32)
    
    # Predict the stock price
    predicted_stock_price = check(regressor, dataset_train, dataset_test, sc)
    
    # Plot the result
    plt.plot(real_stock_price, color = 'red', label = 'Real stock price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted stock price')
    plt.title('Google stock price prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show
    
    # Evaluation: mean squared error
    # Here, we are trying to predict the direction. RMSE does not make much sense
    # unless we divide it by the range of the stock price.
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    print (rmse)
    
""" Main """
if __name__ == "__main__":
    main()