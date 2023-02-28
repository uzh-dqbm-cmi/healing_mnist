"""
Data loader for the Healing MNIST data set (c.f. https://arxiv.org/abs/1511.05121)

Adapted from https://github.com/Nikita6000/deep_kalman_filter_for_BM/blob/master/healing_mnist.py
"""


import numpy as np
import scipy.ndimage
import torchvision
import random


def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img


def apply_noise(img, bit_flip_ratio):
    img = np.array(img)
    mask = np.random.random(size=(28, 28)) < bit_flip_ratio
    img[mask] = 255 - img[mask]
    return img


def get_rotations(img, rotation_steps):
    # yield image for eah rotation step
    for rot in rotation_steps:
        img = scipy.ndimage.rotate(img, rot, reshape=False)
        yield img


def binarize(img):
    return (img > 127).astype(int)


def heal_image(img, seq_len, square_count, square_size, noise_ratio, max_angle):
    squares_begin = np.random.randint(0, seq_len - square_count + 1)
    squares_end = squares_begin + square_count

    rotations = []
    rotation_steps = np.random.normal(size=seq_len, scale=max_angle)

    for idx, rotation in enumerate(get_rotations(img, rotation_steps)):
        # Don't add the squares right now
        if idx >= squares_begin and idx < squares_end:
            rotation = apply_square(rotation, square_size)

        # Don't add noise for now
        # noisy_img = apply_noise(rotation, noise_ratio)
        noisy_img = rotation
        #binarized_img = binarize(noisy_img)
        #rotations.append(binarized_img)

        rotations.append(rotation)

    return rotations, rotation_steps


class HealingMNIST():

    def __init__(self, min_seq_len=1, max_seq_len=5, square_count=3, square_size=5, noise_ratio=0.15, digits=range(10), max_angle=180):
        """Instantiate HealingMNIST() data class

        Args:
            min_seq_len (int, optional): Minimum length for each sequence of digits. Defaults to 1.
            max_seq_len (int, optional): Maximal length for sequence of digits. Defaults to 5.
            square_count (int, optional): Maximal number of squares in a sequence. Defaults to 3.
            square_size (int, optional): Square size. Defaults to 5.
            noise_ratio (float, optional): Injected noise (unused for now). Defaults to 0.15.
            digits (_type_, optional): Digits to use from the dataset. Defaults to range(10).
            max_angle (int, optional): Max rotation. Defaults to 180.
        """
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True)
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
        train_rotations = []
        test_rotations = []
        train_labels = []
        test_labels = []

        for index, (img, label) in enumerate(mnist_train):
            # random sequence length
            seq_len_rand = random.choice(range(min_seq_len, max_seq_len + 1))
            adapted_square_count = random.choice(
                range(0, min(seq_len_rand, square_count)))
            train_img, train_rot = heal_image(
                img, seq_len_rand, adapted_square_count, square_size, noise_ratio, max_angle)
            train_images.append(train_img)
            train_rotations.append(train_rot)
            train_labels.append(label)

        for img, label in mnist_test:
            # random sequence length
            seq_len_rand = random.choice(range(min_seq_len, max_seq_len + 1))
            adapted_square_count = random.choice(
                range(0, min(seq_len_rand, square_count)))
            test_img, test_rot = heal_image(
                img, seq_len_rand, adapted_square_count, square_size, noise_ratio, max_angle)
            test_images.append(test_img)
            test_rotations.append(test_rot)
            test_labels.append(label)

        self.train_images = np.array(train_images)
        self.test_images = np.array(test_images)
        self.train_rotations = np.array(train_rotations)
        self.test_rotations = np.array(test_rotations)
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)


if __name__ == "__main__":
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True)
    d = HealingMNIST()
