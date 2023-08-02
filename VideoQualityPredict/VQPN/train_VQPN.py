import os
import csv
import h5py
import torch
import numpy as np
from utils import get_default_device
from torch.utils.data.dataloader import DataLoader
from vqpn_pt import VQPNModel
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_w = 64
input_h = 36
input_d = 3
input_seq = 25
output_dim = 5
kernel = int(64)
dense_size = int(64)
lr_rate = float(1e-4)
epochs = 1500
batch_size = 50
earlystop = 50

def train_model(lr):
    h5f = h5py.File('./data/train_hd.h5', 'r')
    x = DataLoader(h5f['X'], batch_size)
    y = DataLoader(h5f['Y'], batch_size)
    train_len = 0
    for i in x:
        train_len += 1
    data = {}
    for n in range(train_len):
        data[n] = []
    numx = 0
    for batch_x in x:
        data[numx].append(batch_x)
        numx += 1
    numy = 0
    for batch_y in y:
        data[numy].append(batch_y)
        numy += 1
    h5f_test = h5py.File('./data/test_hd.h5', 'r')
    x_test = h5f_test['X']
    y_test = h5f_test['Y']
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)

    model = VQPNModel(64, 64, 64, 36, 3, 25, 5)
    optconv = torch.optim.SGD([
        {'params':model.conv1.weight},
        {'params':model.conv2.weight}
    ],lr=0.00001, weight_decay=0.0001)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _min_mape, _min_step = 100.0, 0
    sum_loss = 0
    test_sum_acc = 0
    avg_loss_list = ['loss']
    for epoch in range(1, epochs + 1):
        print('\n')
        print('Epoch: ' + str(epoch))
        for num in range(train_len):
            check0 = torch.isnan(data[num][0].to(device)).any()
            if check0.item():
                continue
            predvmaf = model(data[num][0].to(device))
            lable = data[num][1].to(device)
            loss_fun = torch.nn.MSELoss()
            loss = loss_fun(predvmaf.float(), lable.float())
            print("Batch: " + str(num) + " " + "train_loss: " + str(loss.item()))
            sum_loss += loss.item()
            optconv.zero_grad()
            opt.zero_grad()
            loss.backward()
            optconv.step()
            opt.step()
        predvmaf_test = model(x_test.to(device))
        test_lable = y_test.to(device)
        test_loss_fun = torch.nn.MSELoss()
        test_loss = test_loss_fun(predvmaf_test.float(), test_lable.float())
        test_acc = torch.sqrt(test_loss)
        print('Epoch ' + str(epoch) + ': rmse: ' + str(test_acc))
        avg_loss = sum_loss/train_len
        avg_loss_list.append(avg_loss)
        sum_loss = 0

        if _min_mape > test_acc:
            _min_mape = test_acc
            _min_step = epoch
            torch.save(model.state_dict(), "Model/vqpn.pt")
            print('new record')
        else:
            if epoch - _min_step > earlystop:
                print('early stop')
                headers = [' ']
                for i in range(len(avg_loss_list)):
                    epoch = 'epoch_' + str(i + 1)
                    headers.append(epoch)
                with open('/plots/train_loss.csv', 'w') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    f_csv.writerow(avg_loss_list)
                train_epoches = len(avg_loss_list) - 1
                x1 = range(1, train_epoches + 1)
                y1 = avg_loss_list
                y1.remove('loss')
                plt.cla()
                plt.title('Train loss vs. epoch', fontsize = 20)
                plt.plot(x1, y1, '.-')
                plt.xlabel('epoch', fontsize = 20)
                plt.ylabel('Train loss', fontsize = 20)
                plt.grid()
                plt.savefig("plots/Train_loss.png")
                plt.show()
                return

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_model(0.0001)



