# AutoEncoder

# Import the libraries
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the dataset - Just for visualization
def visualize_data(path):
    
    movies = pd.read_csv(path + '/movies.dat',
                         sep='::',
                         header = None,
                         engine = 'python',
                         encoding = 'latin-1')
    
    users = pd.read_csv('ml-1m/users.dat',
                         sep='::',
                         header = None,
                         engine = 'python',
                         encoding = 'latin-1')
    
    ratings = pd.read_csv('ml-1m/ratings.dat',
                         sep='::',
                         header = None,
                         engine = 'python',
                         encoding = 'latin-1')
    
    return users, movies, ratings


# Preparation of training and test sets
def create_sets(path):
    training_set = pd.read_csv(path + '/u1.base', delimiter = '\t')
    training_set = np.array(training_set, dtype = 'int')
    
    test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
    test_set = np.array(test_set, dtype = 'int')

    # Getting the total number of users and movies
    nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

    training_set_m = convert_to_feature_matrix(training_set, nb_users, nb_movies)
    test_set_m = convert_to_feature_matrix(test_set, nb_users, nb_movies)
    
    # Converting the data into Torch tensors
    training_set = torch.FloatTensor(training_set_m)
    test_set = torch.FloatTensor(test_set_m)

    return training_set, test_set, nb_users, nb_movies, training_set_m, test_set_m


# Creating the matrix of features
def convert_to_feature_matrix(data, nb_users, nb_movies):
    
    feature_matrix = []
    for i in range(nb_users):
        feature_matrix.append(np.zeros(nb_movies).tolist())
        
    for i in range(len(data)):
        user_no = data[i, 0]
        movie_no = data[i, 1]
        rating = data[i, 2]
        feature_matrix[user_no - 1][movie_no - 1] = rating
        
    return feature_matrix


# Creating the architecture of the Stacked AE
class StackedAE(nn.Module):
    
    def __init__(self, nb_movies):
        super(StackedAE, self).__init__()
        #self.nb_movies = nb_movies
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 20)
        self.fc5 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x


# Training the Stacked AE     
def run_training(training_set, nb_users, nb_movies, sae_instance, criterion, optimizer, nb_epochs = 200):
    
    for epoch in range(1, nb_epochs + 1):
        
        train_loss = 0
        s = 0.
        
        for user in range (nb_users):
            
            # Add a dimension to create a batch (batch of a single input vector in this case)
            input = Variable(training_set[user]).unsqueeze(0) 
            
            # Copy of input. Will be used to compare with the predictions
            target = input.clone()
            
            # Check if the current user rated a least 1 movie
            if torch.sum(target.data > 0) > 0:
                
                # Forward pass
                output = sae_instance(input)
                
                # Not running the grad on the target values (saves some computation)
                target.require_grad = False
                
                # In the predictions, set the values of the movies that were not rated, to 0
                output[target == 0] = 0
                
                # Compute the loss. Update the loss of the training, adjusting it with the mean corrector factor.
                # In this case (look at the if condition), we are only looking at movies that have a rating. The 
                # mean corrector factor (nb_movies/nb_non_zero_ratings) will help us get the average error.
                loss = criterion(output, target)
                mean_corrector = nb_movies/float(torch.sum(target.data > 0 + 1e-10))
                train_loss += np.sqrt(loss.item()*mean_corrector)
                s += 1.
                
                # Perform a backward pass, and update the weights.
                loss.backward()
                optimizer.step()
                
        print('epoch: ' + str(epoch) + ' s: ' + str(s) + ' loss: ' + str(train_loss.item()/s))

    # Save the model
    torch.save(sae_instance.state_dict(), 'ae_model_trained_5layers.pt')

  
# Testing the SAE
def run_test(training_set, test_set, nb_users, nb_movies, sae_instance, criterion, optimizer):
    
    test_loss = 0
    s = 0.
    output_numpy_all = np.empty([0,nb_movies])
    
    for user in range (nb_users):
        
        # Add a dimension to create a batch (batch of a single input vector in this case)
        input = Variable(training_set[user]).unsqueeze(0) 
        # Target will contain the real ratings (from test_set)
        target = Variable(test_set[user]).unsqueeze(0)
        
        # Check if the current user rated a least 1 movie
        if torch.sum(target.data > 0) > 0:
            
            # Forward pass
            output = sae_instance(input)
            
            # Not running the grad on the target values
            target.require_grad = False
            
            # In the predictions, set the values of the movies that were not rated, to 0
            output[target == 0] = 0
            
            # Add the predictions to the numpy array
            output_numpy = output.data.numpy()
            output_numpy_all = np.vstack((output_numpy_all, output_numpy) )
            
            # Compute the loss. Update the loss of the training, adjusting it with the mean corrector factor.
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0 + 1e-10))
            test_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.

    return output_numpy_all, test_loss.item()/s

  
def single_prediction(model, training_set, user_id, movie_id):
    
    target_user_id = user_id
    target_movie_id = movie_id
    input = Variable(training_set[target_user_id - 1]).unsqueeze(0)
    output = model(input)
    print (output.data.numpy[0, target_movie_id - 1])
    
      
def main():
    
    #path_visualisation = 'ml-1m'   
    """
    users, movies, ratings = visualize_data(path_visualisation)
    """
    
    # Create the sets + variables
    path_sets = 'ml-100k'
    training_set, test_set, nb_users, nb_movies, trainM, testM = create_sets(path_sets)
    
    # Create instance of StackedAE class + criterion + optimizer
    sae = StackedAE(nb_movies)
    criterion = nn.MSELoss()      
    optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
    
    # Check if the model has already been trained
    model_file = 'ae_model_trained_5layers.pt'
    if os.path.exists(model_file):
        # Load the model
        print ('Loading the model from {0}/{1}'.format(os.getcwd(), model_file))
        sae.load_state_dict(torch.load(model_file))
    else :        
        # Run the training
        print ('Training the model')
        run_training(training_set, nb_users, nb_movies, sae, criterion, optimizer, nb_epochs = 200)
        
    # Run the test and get the loss (training set vs test set)
    predictions_numpy, loss = run_test(training_set, test_set, nb_users, nb_movies, sae, criterion, optimizer)
    print(loss)
    
    # Get the prediction for a single user and movie
    """
    single_prediction(sae, training_set, 14, 7)
    """
    
""" Main """
if __name__ == "__main__":
    main()