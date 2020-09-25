import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
import torch.optim as optim

import net
import data_loader
import train
import test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_loader, test_loader = data_loader.loader()

    model = net.Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        train.train(model, device, train_loader, optimizer, epoch)

    PATH = './CNNModel.pt'
    torch.save(model.state_dict(), PATH)

    model = net.Net().to(device)
    model.load_state_dict(torch.load(PATH))
    test.test(model, device, test_loader)

if __name__ == '__main__':
    main()
