# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros(None)
        self.db = np.zeros(None)

        self.momentum_W = np.zeros(None)
        self.momentum_b = np.zeros(None)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        W = self.W
        b = self.b
        
        self.x = x
        
        # Alternate way to compute 
        #Wx = np.dot(W.T, x.T).T
        
        Wx = np.dot(x, W) 
        return (Wx + b)

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        W = self.W
        b = self.b 
        x = self.x 
        
        n = x.shape[0]  
        
        # dL/dW = x.T * (dL/dY) : Ref: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        dW = (1 / n) * np.dot(x.T, delta)
        
        
        db = (1 / n) * np.sum(delta, axis=0, keepdims=True)
        self.dW = dW
        self.db = db
        
        # dL/dx = (dL/dY) * W.T  : Ref: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        dx = (1 / n) * np.dot(delta, W.T)
        return dx
