##### Functions to create data set with square --> can later be extended to create dataset with rotations as well

import numpy as np
import torch 
import matplotlib.pyplot as plt
import sklearn
import torchvision
from torchvision import datasets
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import pandas as pd
import random
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm
import wandb
from torch.utils.data import TensorDataset
import scipy.ndimage

def apply_square(img, square_size):
    #img = np.array(img)
    img[:square_size, :square_size] = 255
    return img

def apply_square_random_size(img, square_size_min=3, square_size_max=10):
    square_size = random.randint(square_size_min, square_size_max)
    img[:square_size, :square_size] = 255
    return img

def rotate_image(img, angle):
    rotated_img = scipy.ndimage.rotate(img, angle, reshape=False)
    return rotated_img

def normalize(img): 
    img = (img) / 255
    return img


class HealingMNIST():

    def __init__(self,  min=3, max=8, ratio=0.15, digits=range(10)):
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True)

        digits = range(10)
        x_train = trainset.data
        y_train = trainset.targets
        x_test = testset.data
        y_test = testset.targets
        mnist_train = [(img, label) for img, label in zip(
            x_train, y_train) if label in digits]
        mnist_test = [(img, label)
                        for img, label in zip(x_test, y_test) if label in digits]
        
        train_images = []
        test_images = []

        train_labels = []
        test_labels = []

        train_squares = []
        test_squares = []
        
        ### for training dataset

        for index, (img, label) in enumerate(mnist_train):
            img = np.array(img)
            if random.uniform(0,1) >= ratio:
                img = apply_square_random_size(img, min, max)
                train_images.append(img)
                train_labels.append(label)
                train_squares.append(1)
            else: 
                train_images.append(img)
                train_labels.append(label)
                train_squares.append(0)
        ## for test data set
        for index, (img, label) in enumerate(mnist_test):
            img = np.array(img)
            if random.uniform(0,1) >= ratio:
                img = apply_square_random_size(img, min, max)
                test_images.append(img)
                test_labels.append(label)
                test_squares.append(1)
            else: 
                test_images.append(img)
                test_labels.append(label)
                test_squares.append(0)
        
        self.train_images = np.array(train_images)
        self.test_images = np.array(test_images)
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)
        self.train_squares = np.array(train_squares)
        self.test_squares = np.array(test_squares)

def imageToTensor(images, labels, squares):
    img_tensor = torch.tensor(images.tolist(), dtype=torch.float32)
    img_tensor = img_tensor.reshape(-1, 1, 28, 28)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    square_tensor = torch.tensor(squares, dtype=torch.long)


    dataset = TensorDataset(img_tensor, label_tensor, square_tensor)
    return dataset


class HealingMNIST_rot_square():

    def __init__(self,  min=3, max=8, ratio_rot=0.5, ratio_square=0.5, digits=range(10)):
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True)

        digits = range(10)
        x_train = trainset.data
        y_train = trainset.targets
        x_test = testset.data
        y_test = testset.targets
        mnist_train = [(img, label) for img, label in zip(
            x_train, y_train) if label in digits]
        mnist_test = [(img, label)
                        for img, label in zip(x_test, y_test) if label in digits]
        
        train_images = []
        test_images = []

        train_labels = []
        test_labels = []

        train_squares = []
        test_squares = []
        
        ### for training dataset

        for img, label in mnist_train:
             # Generate a random number between 0 and 1
            
            img = np.array(img, dtype=float)
            rand_num = np.random.uniform()
        
            # If the random number is less than the rotation threshold, rotate the image
            if rand_num < ratio_rot:
                
                angle = np.random.randint(0, 180)
                rotated_img = rotate_image(img, angle)

                if random.uniform(0,1) >= ratio_square:
                    rotated_img = apply_square_random_size(rotated_img, square_size_min=min, square_size_max=max)
                    train_squares.append(1)
                else: 
                    train_squares.append(0)
                rotated_img = normalize(rotated_img)
                train_images.append(rotated_img)
                train_labels.append(label)
                
            # Otherwise, add the non-rotated image to the nonrotated list
            else:
                if random.uniform(0,1) >= ratio_square:
                    img = apply_square_random_size(img, square_size_min=min, square_size_max=max)
                    train_squares.append(1)
                else: 
                    train_squares.append(0)
                img = normalize(img)
                train_images.append(img)
                train_labels.append(label)
        ## for test data set
        for img, label in mnist_test:
            # Generate a random number between 0 and 1
            
            img = np.array(img, dtype=float)
            rand_num = np.random.uniform()
        
            # If the random number is less than the rotation threshold, rotate the image
            if rand_num < ratio_rot:
                angle = np.random.randint(0, 180)
                rotated_img = rotate_image(img, angle)

                if random.uniform(0,1) >= ratio_square:
                    rotated_img = apply_square_random_size(rotated_img, square_size_min=min, square_size_max=max)
                    test_squares.append(1)
                else: 
                    test_squares.append(0)

                rotated_img = normalize(rotated_img)
                test_images.append(rotated_img)
                test_labels.append(label)
                
            # Otherwise, add the non-rotated image to the nonrotated list
            else:
                if random.uniform(0,1) >= ratio_square:
                    img = apply_square_random_size(img, square_size_min=min, square_size_max=max)
                    test_squares.append(1)
                else: 
                    test_squares.append(0)
                img = normalize(img)
                test_images.append(img)
                test_labels.append(label)
                
        
        self.train_images = np.array(train_images)
        self.test_images = np.array(test_images)
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)
        self.train_squares = np.array(train_squares)
        self.test_squares = np.array(test_squares)