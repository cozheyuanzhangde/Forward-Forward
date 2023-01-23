import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from utils import *

class FFNet(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.num_epochs = 1000
        self.layers = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for d in range(len(dims) - 1):
            self.layers += [FFLayer(dims[d], dims[d + 1]).cuda()]

    """
    There are two approaches for batch training:
    1. Iterate batches for all layers. ---> easy
    2. Iterate batches for each layer. ---> need to create new batches for next layer input
    We use 1 for the following two training methods.
    """

    def train_1(self, data_loader):
        """
        Train method 1: train all layers for each epoch.
        """
        for batch_i, (x_batch, y_batch) in enumerate(data_loader):
            print("Training Batch #", batch_i, '/', len(data_loader))
            batch_pos, batch_neg = create_data_pos(x_batch, y_batch), create_data_neg(x_batch, y_batch)
            batch_pos, batch_neg = batch_pos.to(self.device), batch_neg.to(self.device)
            for epoch in range(self.num_epochs):
                h_batch_pos, h_batch_neg = batch_pos, batch_neg
                for layer_i, layer in enumerate(self.layers):
                    h_batch_pos, h_batch_neg = layer.train(h_batch_pos, h_batch_neg)

    def train_2(self, data_loader):
        """
        Train method 2: train all epochs for each layer.
        """
        for batch_i, (x_batch, y_batch) in enumerate(data_loader):
            print("Training Batch #", batch_i, '/', len(data_loader))
            h_batch_pos, h_batch_neg = create_data_pos(x_batch, y_batch), create_data_neg(x_batch, y_batch)
            h_batch_pos, h_batch_neg = h_batch_pos.to(self.device), h_batch_neg.to(self.device)
            for layer_i, layer in enumerate(self.layers):
                for epoch in range(self.num_epochs):
                    h_batch_pos_epoch, h_batch_neg_epoch = layer.train(h_batch_pos, h_batch_neg)
                h_batch_pos, h_batch_neg = h_batch_pos_epoch, h_batch_neg_epoch

    @torch.no_grad()
    def predict(self, data_loader):
        all_predictions = torch.Tensor([])
        all_labels = torch.Tensor([])
        for batch_i, (x_batch, y_batch) in enumerate(data_loader):
            print("Evaluation Batch #", batch_i, '/', len(data_loader))
            goodness_per_label_batch = []
            for label in range(10):
                h_batch = overlay_labels_on_images(x_batch, label)
                goodness_batch = []
                for layer in self.layers:
                    h_batch = layer(h_batch)
                    goodness_batch += [h_batch.pow(2).mean(1)]
                goodness_per_label_batch += [sum(goodness_batch).unsqueeze(1)]
            goodness_per_label_batch = torch.cat(goodness_per_label_batch, 1)
            predictions_batch = goodness_per_label_batch.argmax(1)
            all_predictions = torch.cat((all_predictions, predictions_batch), 0)
            all_labels = torch.cat((all_labels, y_batch), 0)
        return all_predictions.eq(all_labels).float().mean().item()


class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
        self.opt.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()