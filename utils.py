import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def MNIST_loaders(train_batch_size=1000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    eval_train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    eval_test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, eval_train_loader, eval_test_loader

def create_data_pos(images, labels):
    return overlay_labels_on_images(images, labels)

def create_data_neg(images, labels):
    labels_neg = labels.clone()
    for idx, y in enumerate(labels):
        all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        all_labels.pop(y.item()) # remove y from labels to generate negative data
        labels_neg[idx] = torch.tensor(np.random.choice(all_labels)).cuda()
    return overlay_labels_on_images(images, labels_neg)

def overlay_labels_on_images(images, labels):
    """Replace the first 10 pixels of images with one-hot-encoded labels
    """
    num_images = images.shape[0]
    data = images.clone()
    data[:, :10] *= 0.0
    data[range(0,num_images), labels] = images.max()
    return data

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()