import numpy as np
import struct
import sys
import time

class MNIST_Data_Loader():
    def __init__(self):
        self.training_data_labels_file = open("/Users/sanjay/Downloads/archive/train-labels.idx1-ubyte", "rb")
        self.training_data_images_file = open("/Users/sanjay/Downloads/archive/train-images.idx3-ubyte", "rb")
        self.test_data_labels_file = open("/Users/sanjay/Downloads/archive/t10k-labels.idx1-ubyte", "rb")
        self.test_data_images_file = open("/Users/sanjay/Downloads/archive/t10k-images.idx3-ubyte", "rb")
        #self.training_data_labels = self.unpack(self.training_data_labels_file, 2049)
        #self.training_data_images = self.unpack(self.training_data_images_file, 2051)
        self.test_data_labels = self.unpack(self.test_data_labels_file, 2049)
        #self.test_data_images = self.unpack(self.test_data_images_file, 2051)

        self.training_data_labels_file.close()
        self.training_data_images_file.close()
        self.test_data_labels_file.close()
        self.test_data_images_file.close()

    def unpack(self, name, expected_MSB):
        MSB, num_items = struct.unpack(">II",name.read(8))
        if(MSB != expected_MSB):
            raise ValueError(f"Expected {expected_MSB}, recieved {MSB}")
        if expected_MSB == 2051:
            num_rows, num_cols = struct.unpack(">II", name.read(8))
            print(f"number of rows: {num_rows}\nnumber of cols: {num_cols}")
            array =  name.read()
            images = np.array([[[[array[num_rows*num_cols*num_image + j * 28 + i]] for i in range(num_cols)] for j in range(num_rows)] for num_image in range(num_items)])
            return images
        testdi = name.unpack()
        print(testdi)
        return testdi

m = MNIST_Data_Loader()