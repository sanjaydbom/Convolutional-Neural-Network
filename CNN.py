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

    
    def update_parameters(self,mini_data,learning_rate):
        delta_kernal_layer_one = np.zeros(4,5,5) 
        delta_bias_one = np.zeros(4)
        delta_kernal_layer_two = np.zeros(3,5,5)
        delta_bias_two = np.zeros(4)
        delta_weights = np.zeros(10,192)
        delta_biases = np.zeros(10)
        for image, label in mini_data:
            temp_dklo, temp_bo, temp_dklt, temp_bt, temp_dw, temp_db = self.backprop(image,label)
            delta_kernal_layer_one = delta_kernal_layer_one + temp_dklo
            delta_bias_one = delta_bias_one + temp_bo
            delta_kernal_layer_two = delta_kernal_layer_two + temp_dklt
            delta_bias_two = delta_bias_two + temp_bt
            delta_weights = delta_weights + temp_dw
            delta_biases = delta_biases + temp_db

        x = learning_rate / (2 * len(mini_data))
        delta_kernal_layer_one = delta_kernal_layer_one * x
        delta_bias_one = delta_bias_one * x
        delta_kernal_layer_two = delta_kernal_layer_two * x
        delta_bias_two = delta_bias_two * x
        delta_weights = delta_weights * x
        delta_biases = delta_biases * x

        self.kernal_layer_one = self.kernal_layer_one + delta_kernal_layer_one
        self.bias_one = self.bias_one + delta_bias_one
        self.kernal_layer_two = self.kernal_layer_two + delta_kernal_layer_two
        self.bias_two = self.bias_two + delta_bias_two
        self.weights = self.weights + delta_weights
        self.biases = self.biases + delta_biases

    def backprop(self,image,label):
        label_2 = np.zeros(10)
        label_2[label] = 1
        activation = image
        activations = [activation]
        z = [self.convolute(image,self.kernal_layer_one) + self.bias_one]
        activations.append(self.ReLU(z[-1]))

        z.append(self.ReLU(image.dot(self.weights) + self.biases))
        activations.append(z)
        cost = 2 * (activations[-1] - label_2)
        delta_bias = cost
        delta_weights = activation[-2] * cost * self.derivative(z)
        error = (self.weights * self.derivative(z)).dot(cost)

    def convolute(image, filter):
        temp = np.zeros((len(filter),len(image)-len(filter[0])+1,len(image[0])-len(filter[0][0])+1))
        for i in range(len(temp)):
            for j in range(len(temp[0])):
                for k in range(len(filter)):
                    temp[k][i][j] = np.sum(image[i:i+len(filter[0]),j:j+len(filter[0][0])] * filter[k])
        return temp

    def pooling(self,image):
        temp = np.zeros(np.shape(image))
        for i in range(0,len(temp[0])-1,2):
            for j in range(0, len(temp[0][0])-1,2):
                for k in range(len(temp)):
                    temp[k][i][j] = np.max(image[k][i:i+2, j: j+2])

    def ReLU(self, array):
        return np.maximum(array, 0)

    def derivative(self,array):
        return np.where(array > 0, 1, 0)



    def ReLU(self, array):
        return np.maximum(array, 0)
    






c = CNN
