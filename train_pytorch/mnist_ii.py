import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

# from torchvision.transform import v2
import torchvision
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import os
import json
import numpy as np
import time



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
class MnistDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data_list = []
        self.targets = []


        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = dir_list
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                continue

            cls = path_dir.split(os.sep)[-1]
            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                sample = np.array(Image.open(file_path).resize((28, 28)), dtype=np.float32)
                self.data_list.append(sample)
                self.targets.append(self.class_to_idx[cls])


        self.data_list = torch.tensor(np.array(self.data_list), dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]

        target = self.targets[index]
        return sample, target


def get_data():
    path_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'mnist\testing')
    path_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'mnist\training')


    train_dataset = MnistDataset(path_train)
    test_dataset = MnistDataset(path_test)

    train_data, val_data = random_split(train_dataset, [0.8, 0.2])

    train_loader = DataLoader(train_data, batch_size=64,shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_Loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_Loader
class my_model(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.layer1 = nn.Linear(input,128)
        self.layer2 = nn.Linear(128,output)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        out = self.layer2(x)

        return out

train_data,val_data,test_data = get_data()

print(len(train_data))
print(len(val_data))
print(len(test_data))

start_time = time.time()
model = my_model(784,10).to(device)

LearningRate = 0.0001
EPOHS = 1
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = LearningRate)



run_train_loss = []
run_val_loss = []
accuracy_train =[]
accuracy_val =[]

for i in range(EPOHS):

    model.train()


    correct = 0
    total = 0
    for x, target in train_data:

        x = x.reshape(-1,28*28).to(device)
        target = target.reshape(-1)
        target = torch.eye(10)[target].to(device).to(torch.float32)


        pred = model(x)




        correct += (pred.argmax(dim =1) == target.argmax(dim = 1)).sum().item()
        total += target.size(0)


        loss = loss_fn(pred,target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_train_loss.append(loss.item())
        mean_train_loss = sum(run_train_loss)/ len(run_train_loss)

        accuracy_train.append(correct/total)



    model.eval()


    correct =0
    total =0
    with torch.no_grad():
        for x, target in val_data:
            x = x.reshape(-1,28*28).to(device)

            target = target.reshape(-1).to(torch.int64).to(device)

            pred = model(x)

            predict = torch.argmax(pred,dim=1)
            correct+= (predict == target).sum().item()
            total += target.size(0)

            loss = loss_fn(pred,target)



            run_val_loss.append(loss.item())
            mean_val_loss = sum(run_val_loss)/len(run_val_loss)


end_time = time.time()

print(f"Elapsed time {end_time - start_time}")

plt.figure(figsize=(8, 5))
plt.plot(run_train_loss, label="Train Loss", color="blue")
plt.plot(run_val_loss, label="Validation Loss", color="orange")
plt.title(f"Training and Validation Loss learning {LearningRate}")  # Заголовок графика
plt.xlabel("Iterations")  # Подпись оси X
plt.ylabel("Loss")  # Подпись оси Y
plt.legend()  # Легенда
plt.grid()  # Включаем сетку для удобства
plt.show()

# График для accuracy
plt.figure(figsize=(8, 5))
plt.plot(accuracy_train, label="Train Accuracy", color="blue")
plt.plot(accuracy_val, label="Validation Accuracy", color="orange")
plt.title(f"Training and Validation Accuracy Learning {LearningRate}")  # Заголовок графика
plt.xlabel("Epochs")  # Подпись оси X
plt.ylabel("Accuracy")  # Подпись оси Y
plt.legend()  # Легенда
plt.grid()  # Включаем сетку для удобства
plt.show()