import random
import MNIST_Data_Loader
import numpy as np

class Neural_Network():
    def __init__(self, sizes):
        self.weight = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
        self.bias = [np.random.randn(x,1) for x in sizes[1:]]
        self.size = sizes

    def relu(self, array):
        return (1.0 / (1.0 + np.exp(-array)))
    
    def derivative(self, array):
        return self.relu(array) * (1 - self.relu(array))
    
    def softmax(self,array):
        temp = np.zeros_like(array)
        exp = np.exp(array)
        sum = np.sum(exp)
        for i in range(np.size(temp)):
            temp[i] = exp[i] / sum
        return temp
    
    def feedforward(self, array):
        self.zs = []
        array = array/127.5 - 1
        self.activations = [array]
        for w,b in zip(self.weight,self.bias):
            self.zs.append(np.dot(w,self.activations[-1]) + b)
            self.activations.append(self.relu(self.zs[-1]))
        
        return self.activations[-1]
    

    def SGD(self, training_data, epochs, learning_rate, training_data_batch_size, test_data = None):
        length = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            batched_data = [training_data[i:i+training_data_batch_size] for i in range(0,length,training_data_batch_size)]
            for batch in batched_data:
                delta_weights = [np.zeros(np.shape(self.weight[i])) for i in range(len(self.weight))]
                delta_bias = [np.zeros(np.shape(self.bias[i])) for i in range(len(self.bias))]

                for image, label in batch:
                    temp = np.zeros((10,1))
                    temp[label][0] = 1
                    label = temp
                    temp_delta_weights = [np.zeros(np.shape(self.weight[i])) for i in range(len(self.weight))]
                    temp_delta_bias = [np.zeros(np.shape(self.bias[i])) for i in range(len(self.bias))]

                    output = self.feedforward(image)

                    delta = (output - label) * self.derivative(self.zs[-1])
                    temp_delta_bias[-1] = delta
                    temp_delta_weights[-1] = np.dot(delta, self.activations[-2].transpose())
                    for i in range(2,len(self.size)):
                        temp_delta_bias[-i] = np.dot(self.weight[-i+1].transpose(), delta) * self.derivative(self.zs[-i])
                        temp_delta_weights[-i] = np.dot(temp_delta_bias[-i], self.activations[-i-1].transpose())
                    
                    delta_weights = [delta_weights[i] + temp_delta_weights[i] for i in range(len(delta_weights))]
                    delta_bias = [delta_bias[i] + temp_delta_bias[i] for i in range(len(delta_bias))]

                self.weight = [self.weight[i] - learning_rate / training_data_batch_size * delta_weights[i] for i in range(len(self.weight))]
                self.bias = [self.bias[i] - learning_rate / training_data_batch_size * delta_bias[i] for i in range(len(self.bias))]

            if(test_data):
                print("Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluate(test_data), len(test_data)))
            else:
                print(f"Epoch {epoch} complete")


    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
m = MNIST_Data_Loader.MNIST_Data_Loader("/Users/sanjay/Downloads/archive/train-labels.idx1-ubyte","/Users/sanjay/Downloads/archive/train-images.idx3-ubyte", "/Users/sanjay/Downloads/archive/t10k-labels.idx1-ubyte", "/Users/sanjay/Downloads/archive/t10k-images.idx3-ubyte")
training_data,test_data = m.get_data()
images, labels = training_data
training_data = list(zip(images,labels))
timages,tlabels = test_data
test_data = list(zip(timages, tlabels))

nn = Neural_Network([784,30,10])
nn.SGD(training_data,30,3.0,10,test_data)
temp = nn.feedforward(timages[0])
print(f"predicted {np.argmax(temp)} actual {tlabels[0]}")