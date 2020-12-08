import mlp
import mnist
import activation as ac
import loss
import numpy as np
import time

# random weight init
def weight_init(x, y):
    return np.random.randn(x, y)

# zero bias init
def bias_init(x):
    return np.zeros((1, x))

#initialize neural parameters
learning_rate = 0.004
momentum = 0.996 #0.956
num_bn_layers= 1
mini_batch_size = 10
epochs = 40


# initialize training, validation and testing data
train, val, test = mnist.load_mnist()

net = mlp.MLP(784, 10, [64, 32], [ac.Sigmoid(), ac.Sigmoid(), ac.Sigmoid()], weight_init, bias_init, loss.SoftmaxCrossEntropy(), learning_rate, momentum, num_bn_layers)



start = time.time()

#training neural network
net.get_training_stats(mnist.load_mnist(), epochs, mini_batch_size)
end = time.time()

print("Training time(sec.) =", end-start)

#testing neural network
accuracy = net.validate(test) * 100.0
print("Test Accuracy: " + str(accuracy) + "%")

#save the model
net.save(str(accuracy) + "_acc_nn_model.pkl")