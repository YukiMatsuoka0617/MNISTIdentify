import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#画像の前処理
class ImgTransform():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # テンソル変換
            transforms.Normalize(mean, std)  # 標準化
        ])

    def __call__(self, img):
        return self.transform(img)

#Datasetクラスを継承
class _3ChannelMnistDataset(data.Dataset):
    def __init__(self, img_data, target, transform):
        #[データ数,高さ,横,チャネル数]に
        self.data = img_data.numpy().transpose((0, 2, 3, 1)) /255.
        self.target = target
        self.img_transform = transform #画像前処理クラスのインスタンス

    def __len__(self):
        #画像の枚数を返す
        return len(self.data)

    def __getitem__(self, index):
        #画像の前処理(標準化)したデータを返す
        img_transformed = self.img_transform(self.data[index])
        return img_transformed, self.target[index]

def loader():
    transform = transforms.Compose([
            transforms.ToTensor()])
    train = torchvision.datasets.MNIST(
        root="data/train", train=True, transform=transform, target_transform=None, download=True)
    test = torchvision.datasets.MNIST(
        root="data/test", train=False, transform=transform, target_transform=None, download=True)

    train_data_resized = train.data.numpy()  #torchテンソルからnumpyに
    test_data_resized = test.data.numpy()

    train_data_resized = torch.FloatTensor(np.stack((train_data_resized,)*3, axis=1))  #RGBに変換
    test_data_resized =  torch.FloatTensor(np.stack((test_data_resized,)*3, axis=1))

    train_dataset = _3ChannelMnistDataset(train_data_resized, train.targets, transform=ImgTransform())
    test_dataset = _3ChannelMnistDataset(test_data_resized, test.targets, transform=ImgTransform())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader,test_loader





