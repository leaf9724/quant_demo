import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
sys.path.append('/home/pc/matrad/leaf/factor/quant_demo/vision_method/AlexNet')
from alexNet import AlexNet
from my_dataloader import MyDataSet
import os
import json
import time

mydataset = MyDataSet(r'/home/pc/matrad/leaf/factor/daily_data/data_processed/greay_picture')
vailddataset = MyDataSet(r'/home/pc/matrad/leaf/factor/daily_data/data_processed/grey_vaild')
train_loader = DataLoader(mydataset, batch_size=64, drop_last=False)
validate_loader = DataLoader(vailddataset, batch_size=64, drop_last=False)


def train_func():
    net = AlexNet(num_classes=2, init_weights=True)

    net.to(device)
    #损失函数:这里用交叉熵
    loss_function = nn.BCELoss()
    #优化器 这里用Adam
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    #训练参数保存路径
    save_path = '/home/pc/matrad/leaf/factor/quant_demo/vision_method/AlexNet/AlexNet.pth'
    #训练过程中最高准确率
    best_acc = 0.0

    #开始进行训练和测试，训练一轮，测试一轮
    for epoch in range(10):
        # train
        net.train()    #训练过程中，使用之前定义网络中的dropout
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter()-t1)

        # validate
        net.eval()    #测试过程中不需要dropout，使用所有的神经元
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training')






