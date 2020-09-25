import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import net
import data_loader


def test(model, device, test_loader):
    model.eval()   # put the model into evaluation mode
    correct = 0
    with torch.no_grad():   # gradients are not needed
        for data, target in test_loader:  # iterate through test data
            data, target = data.to(device), target.to(device)   # shift data to devices
            output = model(data)    # forward pass
            pred = output.max(1, keepdim=True)[1] # get the index of the best prediction
            correct += pred.eq(target.view_as(pred)).sum().item()   # see if target and prediction are the same

    print('\nTest set: Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))
