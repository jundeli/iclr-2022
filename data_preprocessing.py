
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import keras
import matplotlib.pyplot as plt
from numpy import linalg as la



def output_data(train_images, train_labels, test_images, test_labels, number, qubits): # down-sample images and divide datasets in different numbers
    image_size_1 = train_images.shape[1]
    image_size_2 = train_images.shape[2]
    n_train = train_images.shape[0]
    n_test = test_images.shape[0]
    number_train = 0
    number_test = 0

    for i in range(n_train):
        if(train_labels[i] == number):
            number_train = number_train + 1
    for i in range(n_test):
        if(test_labels[i] == number):
            number_test = number_test + 1

    dimension = 2**qubits
    train_images_number_sampled = np.zeros((number_train, dimension, dimension))
    index = 0
    for i in range(n_train):
        # if (i%2000 == 0):
        #     print('the image ', i)
        if (train_labels[i]== number):
            for j in range(dimension):
                for k in range(dimension):
                    train_images_number_sampled[index,j,k] = train_images[i,(int)(image_size_1*j/dimension), (int)(image_size_2*k/dimension)]
            index = index + 1
    test_images_number_sampled = np.zeros((number_test, dimension, dimension))
    index = 0
    for i in range(n_test):
        if (test_labels[i]== number):
            for j in range(dimension):
                for k in range(dimension):
                    test_images_number_sampled[index,j,k] = test_images[i,(int)(image_size_1*j/dimension), (int)(image_size_2*k/dimension)]
            index = index + 1

    for i in range(number_train):
        if(la.norm(train_images_number_sampled[i])==0):
            print("image ", i, " has a norm 0, dimension is ", dimension)
        train_images_number_sampled[i] = train_images_number_sampled[i]/la.norm(train_images_number_sampled[i])
    for i in range(number_test):
        if(la.norm(test_images_number_sampled[i])==0):
            print("image ", i, " has a norm 0, dimension is ", dimension)
        test_images_number_sampled[i] = test_images_number_sampled[i]/la.norm(test_images_number_sampled[i])

    file_number_train = "train_images_"+str(number)+"_"+str(dimension)
    file_number_test = "test_images_"+str(number)+"_"+str(dimension)
 
    np.save(file_number_train+"_normalized", train_images_number_sampled)
    np.save(file_number_test+"_normalized", test_images_number_sampled)
    print("normalized finished, dimension is: ", dimension)
    
    return 0




# mnist_dataset = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()


