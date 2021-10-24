import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class LSTM_embed(nn.Module):
    '''
    Class implementing an LSTM-based NN that predicts a category given a joke
    '''

    def __init__(self, input_size, hidden_dim, output_size):
        super(LSTM_embed, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(input_size, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        
        #Linear maaping between the hidden layer and the output layer (output layer : vector of the same size as the number of categories)
        self.linear = nn.Linear(hidden_dim, output_size)
        
        #Softmax to transform values into probabilities
        self.soft = nn.Softmax(dim=0)
        

    def forward(self, coordinates):
        '''Forward pass
        Input : - coordinates : coordinates in the embedding space of the word
        '''
        out, (hn, cn) = self.lstm1(coordinates.view(1, 1, -1)) #apply the lstm layer
        out, (hn, cn) = self.lstm2(out, (hn, cn)) #apply the lstm layer
        out = out.flatten()
        out = out.squeeze() #squeeze the vector to remove 1-D dimensions for later operations
        out = self.linear(out) #apply the linear mapping
        out = self.soft(out) #apply the softmax operation
        return out
    
    
    def learn(self, X_train, y_train, loss_function, optimizer, num_epochs):
        '''Training process : learn the model
    Input : - X_train : clean text for training jokes
            - y_train : categories
            - model : model to learn : LSTM in our case
            - loss_function : loss function to use
            - optimizer : optimizer to use
            - num_epochs : number of epochs   
        '''
        #define the input and output tensors :
        
        tensor_Y = torch.tensor(list(y_train), dtype=torch.float32)


        #training : for each element of the input tensor 'tensor_X', compare the output of the model and the real chord, and learn
        for i in range(num_epochs):

            running_loss = 0.0 #Loss on the current slice of iterations
            for ind, joke in enumerate(X_train): #for each element of the input tensor
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                
                # Step 2 - compute model predictions
                
                #predictions
                joke = torch.tensor(list(joke), dtype=torch.float)
                prediction = self(joke)
                    
                
                #real category
                real_category = tensor_Y[ind]

                #loss
                loss = loss_function(prediction, real_category) 

                # Step 3 - do a backward pass and a gradient update step
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()

                running_loss += loss.item() #add the loss to the running loss
                
                if ind % 500 == 499: #print every 1000 mini_batches
                    print('Epoch : {}/{}, Iteration : {}/{} , Loss : {}'.format(i + 1, num_epochs, ind +1, len(X_train), running_loss/499))
                    running_loss = 0.0 #reset the running loss, so that the next print is only for the slice of 5000 iterations
                
                
    def compute_predictions(self, X_test):
        '''
        Compute predictions for the testing data
        Input : 
            - X_test: clean text for testing jokes
            
    '''
        #define the input and output tensors :
        tensor_X = torch.tensor(list(X_test), dtype=torch.float)
        
        predictions=[]
        for ind, joke in enumerate(tensor_X):
            #prediction
            prediction = self(joke)
            prediction = prediction.detach().numpy()
            
            #create the one-hot vector
            one_hot = np.zeros(23)
            one_hot[np.argmax(prediction)] = 1
            
            #add prediction to list
            predictions.append(one_hot)
        
        predictions = np.array(predictions)
        return predictions