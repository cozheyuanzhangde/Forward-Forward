import torch
import time
from utils import *
from model import *
    
if __name__ == "__main__":
    torch.manual_seed(42)
    train_loader, eval_train_loader, eval_test_loader = MNIST_loaders()

    net = FFNet([784, 2000, 2000, 2000, 2000])
    
    time_training_start = time.time()
    net.train_3(train_loader)
    time_training_end = time.time()
    training_time = round(time_training_end - time_training_start, 2)

    print(f"Training time: {training_time}s")

    print('train error:', str(round((1.0 - net.predict(eval_train_loader)) * 100, 2)) + '%')

    print('test error:', str(round((1.0 - net.predict(eval_test_loader)) * 100, 2)) + '%')
