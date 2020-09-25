import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


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
print(train_data_resized.size())

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
        self.data = img_data.numpy().transpose((0, 2, 3, 1)) /255
        self.target = target
        self.img_transform = transform #画像前処理クラスのインスタンス

    def __len__(self):
        #画像の枚数を返す
        return len(self.data)

    def __getitem__(self, index):
        #画像の前処理(標準化)したデータを返す
        img_transformed = self.img_transform(self.data[index])
        return img_transformed, self.target[index]

train_dataset = _3ChannelMnistDataset(train_data_resized, train.targets, transform=ImgTransform())
test_dataset = _3ChannelMnistDataset(test_data_resized, test.targets, transform=ImgTransform())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


from torch import nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3)
        self.conv = nn.Conv2d(3, 10, kernel_size=4)
        self.fc1 = nn.Linear(640, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1) #行列を線形処理できるようにベクトルに(view(高さ、横))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = Model()

import tqdm
from torch import optim

# 推論モード
def eval_net(net, data_loader, device="cpu"): #GPUある人はgpuに
    #推論モードに
    net.eval()
    ypreds = [] #予測したラベル格納変数
    for x, y in (data_loader):
        # toメソッドでデバイスに転送
        x = x.to(device)
        y = [y.to(device)]
        # 確率が最大のクラスを予測
        # forwardプロパゲーション
        with torch.no_grad():
            _, y_pred = net(x).max(1)
            ypreds.append(y_pred)
            # ミニバッチごとの予測を一つのテンソルに
            y = torch.cat(y)
            ypreds = torch.cat(ypreds)
            # 予測値を計算(正解＝予測の要素の和)
            acc = (y == ypreds).float().sum()/len(y)
            return acc.item()


# 訓練モード
def train_net(net, train_loader, test_loader,optimizer_cls=optim.Adam, 
              loss_fn=nn.CrossEntropyLoss(),n_iter=4, device="cpu"):
    train_losses = []
    train_acc = []
    eval_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):  #4回回す
        runnig_loss = 0.0
        # 訓練モードに
        net.train()
        n = 0
        n_acc = 0
    
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader),
                                     total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            output = net(xx)
            
            loss = loss_fn(output, yy)
            optimizer.zero_grad()   #optimizerの初期化
            loss.backward()   #損失関数(クロスエントロピー誤差)からバックプロパゲーション
            optimizer.step()
            
            runnig_loss += loss.item()
            n += len(xx)
            _, y_pred = output.max(1)
            n_acc += (yy == y_pred).float().sum().item()
            
        train_losses.append(runnig_loss/i)
        # 訓練データの予測精度
        train_acc.append(n_acc / n)
        # 検証データの予測精度
        eval_acc.append(eval_net(net, test_loader, device))

        # このepochでの結果を表示
        print("epoch:",epoch+1, "train_loss:",train_losses[-1], "train_acc:",train_acc[-1],
              "eval_acc:",eval_acc[-1], flush=True)

eval_net(model, test_loader)
train_net(model, train_loader, test_loader)

data = train_dataset.__getitem__(0)[0].reshape(1, 3, 28, 28) #リサイズ（データローダーのサイズに注意）
print("ラベル",train_dataset.__getitem__(0)[1].data)
model.eval()
output = model(data)
print(output.size())

# モデルの保存
model.eval()
#サンプル入力サイズ
example = torch.rand(1, 3, 28, 28)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./CNNModel.pt")
print(model)