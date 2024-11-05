import numpy as np
import random
from scipy.signal import convolve2d
import MNIST_Data_Loader
import time



kernal_layer_one = np.random.randn(4,5,5)
bias_one = np.random.randn(4)
kernal_layer_two = np.random.randn(3,5,5)
bias_two = np.random.randn(3)
weights = np.random.randn(10,192)
biases = np.random.randn(10)

m = MNIST_Data_Loader.MNIST_Data_Loader("/Users/sanjay/Downloads/archive/train-labels.idx1-ubyte","/Users/sanjay/Downloads/archive/train-images.idx3-ubyte", "/Users/sanjay/Downloads/archive/t10k-labels.idx1-ubyte", "/Users/sanjay/Downloads/archive/t10k-images.idx3-ubyte")
training_data,test_data = m.get_data()
images, labels = training_data
length = len(images)
training_data = list(zip(images,labels))
for epoch in range(30):
    print(epoch)
    random.shuffle(training_data)
    split_training_data = [training_data[k:k+10] for k in range(0,length,10)]
    asdf = 0
    for mini_data in split_training_data:
        print(asdf)
        asdf += 1
        dklo = np.zeros((4,5,5))
        dbo = np.zeros(4)
        dklt = np.zeros((3,5,5))
        dbt = np.zeros(3)
        dw = np.zeros((10,192))
        db = np.zeros(10)
        for image, label in mini_data:
            label_2 = np.zeros(10)
            label_2[label] = 1
            image = np.array(image)
            z1 = np.zeros((4,24,24))
            for i in range(23):
                for j in range(23):
                    for k in range(4):
                        z1[k][i][j] = np.sum(image[i:i+5,j:j+5] * kernal_layer_one[k]) + bias_one[k]
            #relu1
            relu1 = np.maximum(z1, 0)
            pos1 = []
            pool1 = np.zeros((4,12,12))
            for i in range(12):
                for j in range(12):
                    for k in range(4):
                        b = np.argmax(relu1[k][2*i:2*i+2, 2*j: 2*j+2], axis = 0)
                        b = [k,b[0],b[1]]
                        pos1.append(b)
                        pool1[k][i][j] = np.max(relu1[k][2*i:2*i+2, 2*j: 2*j+2])
            z2 = np.zeros((3,4,8,8))
            for i in range(8):
                for j in range(8):
                    for k in range(3):
                        for l in range(4):
                            z2[k][l][i][j] = np.sum(pool1[l][i:i+5,j:j+5] * kernal_layer_two[k]) + bias_two[k]
            relu2 = np.maximum(z2, 0)
            pos2 = []
            pool2 = np.zeros((3,4,4,4))
            for i in range(4):
                for j in range(4):
                    for k in range(3):
                        for l in range(4):
                            b = np.argmax(relu2[k][l][2*i:2*i+2, 2*j: 2*j+2], axis = 0)
                            b = [k,l,b[0],b[1]]
                            pos2.append(b)
                            pool2[k][l][i][j] = np.max(relu2[k][l][2*i:2*i+2, 2*j: 2*j+2])
            unraveled = pool2.ravel()

            z3 = unraveled.dot(np.transpose(weights)) + biases
            activations1 = np.maximum(z3, 0)
            cost = 2 * (label_2 - activations1)
            delta_bias = cost
            derivative = np.where(z3 > 0, 1, 0)
            delta_weights = (unraveled * np.atleast_2d(cost).T) * np.atleast_2d(derivative).T
            error = (weights * np.atleast_2d(derivative).T).T.dot(np.atleast_2d(cost).T)
            error = error.reshape(np.shape(pool2))#3,4,4,4
            delta_bias_two = np.sum(np.sum(np.sum(error,axis=2),axis=1),axis = 1) // 18
            temp_error = np.zeros(np.shape(z2))
            c = 0
            for i in range(3):
                for j in range(4):
                    for k in range(4):
                        for l in range(4):
                            temp_error[pos2[c][0]][pos2[c][1]][pos2[c][2]][pos2[c][3]] = error[i][j][k][l]
            error = temp_error
            temp_delta_kernal_layer_two = np.zeros((3,4,5,5))

            for i in range(3):
                for j in range(4):
                    for k in range(5):
                        for l in range(5):
                            temp_delta_kernal_layer_two[i][j][k][l] = np.sum(pool1[j][k:k+8,l:l+8] * error[i][j])
            delta_kernal_layer_two = np.sum(temp_delta_kernal_layer_two, axis = 1)
            temp_filter_two = np.rot90(kernal_layer_two,2,axes=(1,2))
            temp_error = np.zeros((3,4,12,12))
            for i in range(3):
                for j in range(4):
                    for k in range(12):
                        for l in range(12):
                            temp_error[i][j] = convolve2d(temp_filter_two[i], error[i][j])
            error = np.sum(temp_error, axis = 0)

            temp_error = np.zeros(np.shape(relu1))
            for i in range(4):
                for j in range(12):
                    for k in range(12):
                            temp_error[pos2[c][0]][pos2[c][1]][pos2[c][2]] = error[i][j][k]
            error = temp_error

            delta_bias_one = np.sum(np.sum(error, axis = 1), axis = 1) // 12
            delta_kernal_layer_one = np.zeros((4,5,5))
            for i in range(4):
                for j in range(5):
                    for k in range(5):
                        delta_kernal_layer_one[i][j][k] = np.sum(image[j:j+24,k:k+24] * error[i])

            dklo += delta_kernal_layer_one
            dbo += delta_bias_one
            dklt += delta_kernal_layer_two
            dbt += delta_bias_two
            dw += delta_weights
            db += delta_bias
        x = 10 / (2 * len(mini_data))
        dklo = dklo * x
        dbo = dbo * x
        dklt = dklt * x
        dbt = dbt * x
        dw = dw * x
        db = db * x
    kernal_layer_one += dklo
    bias_one += dbo
    kernal_layer_two += dklt
    bias_two += dbt
    weights += dw
    biases += db



td = training_data[0]
image = td[0]
label = td[1]

print(label)
z1 = np.zeros((4,24,24))
for i in range(23):
    for j in range(23):
        for k in range(4):
            z1[k][i][j] = np.sum(image[i:i+5,j:j+5] * kernal_layer_one[k]) + bias_one[k]
            #relu1
relu1 = np.maximum(z1, 0)
pos1 = []
pool1 = np.zeros((4,12,12))
for i in range(12):
    for j in range(12):
        for k in range(4):
            b = np.argmax(relu1[k][2*i:2*i+2, 2*j: 2*j+2], axis = 0)
            b = [k,b[0],b[1]]
            pos1.append(b)
            pool1[k][i][j] = np.max(relu1[k][2*i:2*i+2, 2*j: 2*j+2])

z2 = np.zeros((3,4,8,8))
for i in range(8):
    for j in range(8):
        for k in range(3):
            for l in range(4):
                z2[k][l][i][j] = np.sum(pool1[l][i:i+5,j:j+5] * kernal_layer_two[k]) + bias_two[k]

relu2 = np.maximum(z2, 0)
pos2 = []
pool2 = np.zeros((3,4,4,4))
for i in range(4):
    for j in range(4):
        for k in range(3):
            for l in range(4):
                b = np.argmax(relu2[k][l][2*i:2*i+2, 2*j: 2*j+2], axis = 0)
                b = [k,l,b[0],b[1]]
                pos2.append(b)
                pool2[k][l][i][j] = np.max(relu2[k][l][2*i:2*i+2, 2*j: 2*j+2])
unraveled = pool2.ravel()

z3 = unraveled.dot(np.transpose(weights)) + biases
activations1 = np.maximum(z3, 0)
print(activations1)

