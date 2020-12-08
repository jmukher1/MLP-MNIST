import numpy as np
from numpy import linalg as la

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """
        if eval:
            x_norm=(x-self.running_mean)/np.sqrt(self.running_var+self.eps)
            self.out=self.gamma*x_norm + self.beta
        else:
            self.x = x
            self.mean=np.mean(x,axis=0)
            self.var = np.var(x,axis=0)
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
            self.norm=(self.x-self.mean)/np.sqrt(self.var+self.eps)
            self.out = self.norm*self.gamma+self.beta
        return self.out

     
    
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        m = self.x.shape[0] 

        dldxhat = delta * self.gamma
        dldvar = np.sum(dldxhat*(self.x-self.mean)*(-1/2)*((self.var+self.eps)**(-3/2)), axis=0)
        dldu = -np.sum(dldxhat*((self.var+self.eps)**(-1/2)), axis=0) - 2*dldvar*np.mean(self.x-self.mean, axis=0)
        dldxi = dldxhat*((self.var+self.eps)**(-1/2)) + dldvar*(2/m)*(self.x-self.mean) + dldu/m
        self.dgamma = np.sum(delta*self.norm, axis=0, keepdims = True)
        self.dbeta = np.sum(delta, axis=0, keepdims = True)
        
        return dldxi