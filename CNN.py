import numpy as np
import random
import MNIST_Data_Loader

class CNN():
    def __init__(self):
        self.kernal_layer_one = np.random.rand(4,5,5)
        self.bias_one = np.random.rand(4)
        self.kernal_layer_two = np.random.rand(3,5,5)
        self.bias_two = np.random.rand(4)
        self.weights = np.random.rand(10,192)
        self.biases = np.random.rand(10)

    def training(self, data):
        length = len(data)
        for i in range(30):
            random.shuffle(data)
            temp = [data[k:k+10] for k in range(0,length,10)]
            for mini_data in temp:
                self.update_parameters(mini_data, 1.0)

    def backprop(self,image,label):
        return 1,1,1,1,1,1
    
    def update_parameters(self,mini_data,learning_rate):
        delta_kernal_layer_one = np.zeros(4,5,5) 
        delta_bias_one = np.zeros(4)
        delta_kernal_layer_two = np.zeros(3,5,5)
        delta_bias_two = np.zeros(4)
        delta_weights = np.zeros(10,192)
        delta_biases = np.zeros(10)
        for image, label in mini_data:
            temp_dklo, temp_bo, temp_dklt, temp_bt, temp_dw, temp_db = backprop(image,label)



    def ReLU(self, array):
        return np.maximum(array, 0)
    






c = CNN
