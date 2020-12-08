"""

"""

import numpy as np
import os
import pickle
import time 
import activation
import numpy as np
import sys

from loss import *
from activation import *
from batchnorm import *
from linear import * 

class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.hiddens = hiddens
        self.nlayers = len(self.hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        
        self.inputs = [0] * (self.nlayers + 1)
        self.activation_inputs = [0] * self.nlayers
        self.shapes = [self.input_size] + hiddens + [self.output_size]
        
        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = []
         
        in_feature_count = input_size
        out_feature_count = output_size
        self.shapes = [self.input_size] + hiddens + [self.output_size]
        
        for i in range(self.nlayers):
            out_feature_count = output_size
            # For final layer out_feature is output, hence out_feature size is 
            if i != (self.nlayers - 1):
                out_feature_count = hiddens[i]
            else:
                out_feature_count = output_size
            linear_layer = Linear(in_feature_count, out_feature_count, weight_init_fn, bias_init_fn)    
            self.linear_layers.append(linear_layer)
            in_feature_count = out_feature_count
        
        
        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = [0] * self.nlayers
        for idx in range(self.nlayers):
            self.W[idx] = weight_init_fn(self.shapes[idx], self.shapes[idx+1])
        self.dW = [0] * self.nlayers
        self.b = [0] * self.nlayers
        for idx in range(self.nlayers):
            self.b[idx] = bias_init_fn(self.shapes[idx+1])
        self.db = [0] * self.nlayers
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        self.bn_layers = []
        if self.bn:
            for i in range(1, self.num_bn_layers+1):
                self.bn_layers.append(BatchNorm(self.shapes[i]))
                
        self.validation_acc = []
        
    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        self.inputs[0] = x
                                  
        for i in range(self.nlayers):
            linear = self.linear_layers[i]
            W = linear.W
            b = linear.b 
            z = np.dot(self.inputs[i],W) + b
            #if i == 0:
                #if i in range(self.num_bn_layers):
                    #z = self.bn_layers[i].forward(z, not self.train_mode)
            if i < self.num_bn_layers:
                z = self.bn_layers[i].forward(z, not self.train_mode)
            self.activation_inputs[i] = z
            self.inputs[i+1] = self.activations[i](z)
        
        return self.inputs[-1]

    def zero_grads(self):
        for idx in range(self.nlayers):
            self.dW[idx] = np.zeros((self.shapes[idx], self.shapes[idx+1]))
        for idx in range(self.nlayers):
            self.db[idx] = np.zeros(self.shapes[idx+1])

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers
        for i in reversed(range(self.nlayers)):
            # Update weights and biases here
            # Here we perform a stochastic gradient descent step. 
            self.linear_layers[i].momentum_W = (self.momentum * self.linear_layers[i].momentum_W) - (self.lr * self.linear_layers[i].dW)
            self.linear_layers[i].momentum_b = (self.momentum * self.linear_layers[i].momentum_b) - (self.lr * self.linear_layers[i].db) 
            self.linear_layers[i].W += self.linear_layers[i].momentum_W
            self.linear_layers[i].b += self.linear_layers[i].momentum_b
            
        # Do the same for batchnorm layers 
        for i in range(len(self.bn_layers)):
            self.bn_layers[i].gamma -=  (self.lr * self.bn_layers[i].dgamma)
            self.bn_layers[i].beta -= (self.lr * self.bn_layers[i].dbeta)
        
      
    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        self.zero_grads()
        m = self.inputs[0].shape[0]
        self.y_loss_criterion = self.loss(labels)
        dy_k = (1/m) * self.criterion.derivative()
        
        for k in reversed(range(self.nlayers)):
            dZ_k = self.activations[k].derivative() * dy_k
            if self.bn and self.train_mode==True:
                #if k == 0:
                    #dZ_k = self.bn_layers[0].backward(dZ_k)
                if k < self.num_bn_layers:
                    dZ_k = self.bn_layers[k].backward(dZ_k)
            
            self.linear_layers[k].dW = np.dot(self.inputs[k].T, dZ_k)
            self.linear_layers[k].db = np.sum(dZ_k, axis=0, keepdims=True)
            dy_k = np.dot(dZ_k, self.linear_layers[k].W.T)
            
    def loss(self, labels):
        return self.criterion(self.inputs[-1], labels)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def get_training_stats(mlp, dset, nepochs, batch_size):
        """Fit (train) the MLP on provided training data.

        Parameters
        ----------
        training_data : array of lists
            [0],[1] = image, label.

        validation_data : array of lists, optional
            If provided, the network will count
            validation accuracy after each epoch.

        nepochs : number of epochs, optional
            By default it equals 40

        batch_size : size of minibatches, optional
            By default it equals 10

        """
        train, val, test = dset
        trainx, trainy = train
        valx, valy = val
        testx, testy = test

        idxs = np.arange(len(trainx))

        training_losses = []
        training_errors = []
        validation_losses = []
        validation_errors = []
        test_losses = []
        test_errors = []
        training_losses_stats = []
        training_errors_stats = []
        validation_losses_stats = []
        validation_errors_stats = []

        np.random.seed(123)
        model = mlp
        model.train()

        for e in range(nepochs):

            # Per epoch setup ...
            seed = np.random.randint(123)
            np.random.seed(seed)
            np.random.shuffle(trainx)
            np.random.seed(seed)
            np.random.shuffle(trainy)

            seed = np.random.randint(123)
            np.random.seed(seed)
            np.random.shuffle(valx)
            np.random.seed(seed)
            np.random.shuffle(valy)

            model.train()

            for b in range(0, len(trainx), batch_size):

                # Train ...
                x_batch = trainx[b:b + batch_size]
                y_batch = trainy[b:b + batch_size]

                model.zero_grads()
                preds = model.forward(x_batch)
                model.backward(y_batch)
                loss = model.y_loss_criterion
                model.step()

                answers = np.argmax(preds, axis=1)
                labels = np.argmax(y_batch, axis=1)
                error = (answers[answers!=labels]).shape[0] / len(answers)

                training_losses_stats.append(loss)
                training_errors_stats.append(error)

            for b in range(0, len(valx), batch_size):

                # Evaluate/Validate ...
                model.eval()

                x_batch = valx[b:b + batch_size]
                y_batch = valy[b:b + batch_size]

                model.zero_grads()
                preds = model.forward(x_batch)
                #print("preds shape = ", preds.shape, ", y_batch shape = ", y_batch.shape)
                loss = model.criterion(preds, y_batch)

                answers = np.argmax(preds, axis=1)
                labels = np.argmax(y_batch, axis=0)
                error = float(len(answers[answers!=labels])) / len(answers)

                validation_losses_stats.append(loss)
                validation_errors_stats.append(error)            


            # Accumulate data...
            training_losses.append(np.mean(training_losses_stats))
            training_errors.append(np.mean(training_errors_stats))

            validation_losses.append(np.mean(validation_losses_stats))
            validation_errors.append(np.mean(validation_errors_stats))
            
            if val:
                accuracy = model.validate(val) * 100.0
                print("Epoch {0}, accuracy {1} %.".format(e + 1, accuracy))
                model.validation_acc.append(accuracy)
            else:
                print("Processed epoch {0}.".format(e))


        # Cleanup ...
        model.eval()

        seed = np.random.randint(123)
        np.random.seed(seed)
        np.random.shuffle(testx)
        np.random.seed(seed)
        np.random.shuffle(testy)    

        for b in range(0, len(testx), batch_size):

            # Test ...
            x_batch = testx[b:b + batch_size]
            y_batch = testy[b:b + batch_size]

            model.zero_grads()
            preds = model.forward(x_batch)
            model.backward(y_batch)
            loss = model.criterion(model.inputs[-1], y_batch)

            answers = np.argmax(preds, axis=1)
            labels = np.argmax(y_batch, axis=0)
            error = len(answers[answers!=labels]) / len(answers)

            test_losses.append(loss)
            test_errors.append(error)

        # Return results ...
        return (training_losses, training_errors, validation_losses, validation_errors)
    
    def validate(self, validation_data):
        """Function uses the
        number of correctly predicted classes as validation accuracy metric.

        Parameters
        ----------
        validation_data : list

        Returns
        -------
        int
            Percent of correctly predicted classes.
        """
        counter = 0
        for idx, x in enumerate(validation_data[0]):
            predicted = self.predict(x)
            #print("actual = ", validation_data[1][idx], " -> predicted = ", predicted)
            if self.predict(x) == validation_data[1][idx]:
                counter += 1

        return counter/len(validation_data[1])

    def predict(self, x):
        """Predict the class of a single test example.

        Parameters
        ----------
        x : numpy.array

        Returns
        -------
        int
            Predicted class.

        """
        self.eval()
        self.forward(x)
        predicted = np.argmax(self.inputs[-1], axis=1)
        return predicted


    def load(self, filename='nn_model.pkl'):
        """Load serialized model with weights and biases

        Parameters
        ----------
        filename : str, optional
        Name of the ``.pkl`` serialized object.

        """
        with open(filename,'rb') as f:
            nn_model = pickle.load(f, encoding='bytes')
        f.close()

        self.W = nn_model.W
        self.b = nn_model.b

        self.num_bn_layers = nn_model.num_bn_layers
        self.bn = nn_model.num_bn_layers > 0
        self.hiddens = nn_model.hiddens
        self.nlayers = len(nn_model.hiddens) + 1
        self.input_size = nn_model.input_size
        self.output_size = nn_model.output_size
        self.activations = nn_model.activations
        self.criterion = nn_model.criterion
        self.lr = nn_model.lr
        self.momentum = nn_model.momentum

        if self.bn:
            self.bn_layers = nn_model.bn_layers

        self.train_mode = nn_model.train_mode
        self.batch_size = nn_model.batch_size
        self.epochs = nn_model.epochs

    def save(self, filename='nn_model.pkl'):
        """Save serialized model of neural network

        Parameters
        ----------
        filename : str, optional
        Name of the ``.pkl`` serialized object

        """
        seconds = time.time()

        directory = os.path.join(os.curdir, 'models')
        filepath = os.path.join(directory, str(seconds)+'_'+filename)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        f.close()