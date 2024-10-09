import numpy as np
import struct
from array import array

class MNIST_Data_Loader():
    def __init__(self, training_data_lable_filepath,training_data_images_filepath,test_data_labels_filepath,test_data_images_filepath):
        self.training_data_labels_file = open(training_data_lable_filepath, "rb")
        self.training_data_images_file = open(training_data_images_filepath, "rb")
        self.test_data_labels_file = open(test_data_labels_filepath, "rb")
        self.test_data_images_file = open(test_data_images_filepath, "rb")

        self.training_data_labels = self.unpack(self.training_data_labels_file, 2049)
        self.training_data_images = self.unpack(self.training_data_images_file, 2051)
        self.test_data_labels = self.unpack(self.test_data_labels_file, 2049)
        self.test_data_images = self.unpack(self.test_data_images_file, 2051)

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
            unordered_data = array("B", name.read())
            images = [0] * num_items
            for i in range(num_items):
                images[i] = np.reshape(unordered_data[i * num_rows * num_cols : (i + 1) * num_rows * num_cols],(28,28))
            return images
        
        labels = np.array(array("B", name.read()))
        return labels
    
    def get_data(self):
        return(self.training_data_images, self.training_data_labels), (self.test_data_images, self.test_data_labels)
