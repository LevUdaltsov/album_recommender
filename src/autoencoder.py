""" 
    This is a script that contains autoencoder objects
"""

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1000, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True), nn.Linear(64, 1000), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AETrainer:


    def __init__(self, model, num_epochs, batch_size, criterion, optimizer, device, model_save_path):
        self.device = device
        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.losses = []


    def train(self, dataset, test_split=0.2):

        train_data, test_data = train_test_split(dataset, test_size=test_split)
        train_loader = DataLoader(torch.from_numpy(train_data), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(torch.from_numpy(test_data), batch_size=self.batch_size, shuffle=True)


        n_train_batches = int(np.floor(len(train_data) / self.batch_size))
        n_test_batches = int(np.floor(len(test_data) / self.batch_size))


        best_loss = np.inf
        i = 0
        for epoch in range(self.num_epochs):

            training_loss = 0
            for item in train_loader:
            
                data = item.float()
                data = data.to(self.device)
                self.optimizer.zero_grad()

                output = self.model(data)

                loss = self.criterion(output, data)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
            training_loss /= n_train_batches
            self.losses.append(training_loss)
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.num_epochs, loss.data))


            test_loss = self._test(test_loader, n_test_batches, epoch)

            if test_loss < best_loss:
                print('Model improved from {} to {}'.format(best_loss, test_loss))
                best_loss = test_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                i = 0
            
            else:
                print('Model did not improve on loss of ' + str(best_loss))
                i+=1
            if i == 20:
                break


    def _test(self, test_loader, n_test_batches, epoch):

        self.model.eval()
        test_loss = 0
        for item in test_loader:
            data = item.float()
            data = data.to(self.device)

            output = self.model(data)

            loss = self.criterion(output, data)

            test_loss += loss.item()
        test_loss /= n_test_batches
       
        return test_loss    