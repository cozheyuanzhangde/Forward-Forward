import torch
from utils import *
from model import *
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = FFNet([784, 2000, 1000, 500, 200, 500, 1000])
    train_images, train_labels = next(iter(train_loader))
    train_images, train_labels = train_images.cuda(), train_labels.cuda()
    data_pos = create_data_pos(train_images, train_labels)
    data_neg = create_data_neg(train_images, train_labels)
    
    # for data, name in zip([train_images, data_pos, data_neg], ['orig', 'pos', 'neg']):
    #     visualize_sample(data, name)
    
    net.train_1(train_loader)

    # print('train error:', 1.0 - net.predict(train_images).eq(train_labels).float().mean().item())

    # test_images, test_labels = next(iter(test_loader))
    # test_images, test_labels = test_images.cuda(), test_labels.cuda()

    # print('test error:', 1.0 - net.predict(test_images).eq(test_labels).float().mean().item())
