# Import the libraries
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix

# BUILD THE ANN
def build_model(dropout_rate = 0.2, optimizer = 'adam'):

    # Initialize the ANN
    model = Sequential()
    
    # Add the input layer + 1st hidden layer
    model.add(Dense(16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dropout(dropout_rate))
    
    # Add a 2nd hidden layer
    model.add(Dense(16, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    
    # Add the final layer
    model.add(Dense(1, kernel_initializer = 'uniform' ,activation = 'sigmoid'))
    
    # Complie the ANN
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


# TRAIN THE ANN
def run_training(model, X_train, y_train, batch_size = 64, epochs = 400):
    
    # Train the ANN
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
    

# PERFORMANCE EVALUATION AND TUNING
def run_kFoldCv(X_train, y_train):
    #  K-Fold Cross Validation
    # Here we built the model with Keras, however the cross val function belongs to scikit-learn
    # So we need to combine keras and sklearn
    # Keras wrapper that will wrap Kfold CV by sklearn into the Keras model
    
    # Wrap the whole thing
    wrappedClassifier = KerasClassifier(build_fn = build_model, batch_size = 10, epochs = 25)
    
    # accuracies will be a vector storing the accuracy of each ANN
    accuracies = cross_val_score(estimator = wrappedClassifier, X = X_train, y = y_train, cv = 10)
    
    return accuracies


def run_gridSearch(X_train, y_train):
    # Tuning the model: Grid Search method
    # Method to tune and test different combinations of parameters/hyperparameters
    
    # Wrap the whole thing
    wrappedClassifier = KerasClassifier(build_fn = build_model(dropout_rate = 0.2, optimizer = 'adam'))
    
    # Create parameters dictionary 
    params = {'batch_size': [10, 20],
              'epochs': [15, 20],
              'Optimizer': ['adam', 'nadam']
              }
    
    gridSearch = GridSearchCV(estimator = wrappedClassifier, 
                              param_grid = params,
                              scoring = 'accuracy',
                              cv = 7)
    
    gridSearch = gridSearch.fit(X_train, y_train)
    
    best_param = gridSearch.best_params_
    best_accuracy = gridSearch.best_score_
    
    return best_param, best_accuracy


def main():
    
    # Pre-processing - Import the dataset with pandas
    dataset = pd.read_csv('Churn_Modelling.csv')
    
    # Pre-processing - Seperate your dataset in X in y
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    
    # Pre-processing - Categorical data. Use LabelEncoder to encode the categorical variables
    le_geo = LabelEncoder()
    X[:, 1] = le_geo.fit_transform(X[:, 1])
    le_gender = LabelEncoder()
    X[:, 2] = le_gender.fit_transform(X[:, 2])
    # Pre-processing - Use OneHotEncoder on the geographical variables
    ohe = OneHotEncoder(categorical_features = [1])
    X = ohe.fit_transform(X).toarray()
    # Pre-processing - Remove the first colum to avoid the dummy trap
    X = X[:, 1:]
    
    # Pre-processing - Standardize the data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    # Pre-processing - Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    # Build the model
    classifier = build_model(dropout_rate = 0.2, optimizer = 'adam')
    # Train the model
    classifier_trained = run_training(classifier, X_train, y_train, batch_size = 64, epochs = 400)
    
    # Run the predictions on X_test
    y_pred = classifier_trained.predict(X_test)
    y_pred = (y_pred > 0.5)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print (cm)
    
    # Run the K-fold cross validation. It will return a vector of all the accuracies
    accuracies = run_kFoldCv(X_train, y_train)
    mean = accuracies.mean()
    var = accuracies.std()
    print (mean, var)
    
    # Run the Grid Search method. It will return the best parameters and the accuracy it obtained using them.
    best_param, best_accuracy = run_gridSearch(X_train, y_train)
    print (best_param, best_accuracy)

""" Main """    
if __name__ == "__main__":
    main()