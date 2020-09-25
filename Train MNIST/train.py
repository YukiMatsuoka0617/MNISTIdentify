import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import net
import data_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()   # put the model into training mode
    for batch_idx, (data, target) in enumerate(train_loader):   # iterate through batches
        data, target = data.to(device), target.to(device)   # shift data to devices
        optimizer.zero_grad()   # set gradients to zero
        output = model(data)    # forward pass
        loss = F.nll_loss(output, target)   # compute the loss
        loss.backward()     # backwar propagation
        optimizer.step()    # next steps
        if batch_idx % 10 == 0:   # let's print it from time to time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
