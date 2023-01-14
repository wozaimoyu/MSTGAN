import argparse
from math import log10
import h5py
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import get_allfile, h5_read_data, floatrange

def h5_read_data(filename, dataset):
    with h5py.File(filename, 'r') as f:
        dset = f[dataset]
        return dset[:]

def get_dataset(path_val, path_test,model_path,device,batch_size,slot=2,T=5):  # 获取所有文件

    with torch.no_grad():
        net_g = torch.load(model_path,map_location='cuda:0').to(device)
    y_val_mse = []
    y_val_nmse =[]
    y_test_mse = []
    y_test_nmse =[]

    all_file_val = get_allfile(path_val)  # tickets要获取文件夹名
    all_file_test = get_allfile(path_test)  # tickets要获取文件夹名
    all_file_val.sort()
    all_file_test.sort()
    print(all_file_val)
    print(all_file_test)
    # gt
    perfect = []
    video = []
    perfect_test = []
    video_test = []
    for i_f in range(len(all_file_val)):
        print(i_f)
        val_data = h5_read_data(all_file_val[i_f], 'ls')
        val_label = h5_read_data(all_file_val[i_f], 'gt')
        test_data = h5_read_data(all_file_test[i_f], 'ls')
        test_label = h5_read_data(all_file_test[i_f], 'gt')

        val_data = torch.from_numpy(val_data[:, :, T - slot:T:1, :, :])
        val_label = torch.from_numpy(val_label)
        test_data = torch.from_numpy(test_data[:, :, T - slot:T:1, :, :])
        test_label = torch.from_numpy(test_label)

        val_dataset = TensorDataset(val_data, val_label)
        test_dataset = TensorDataset(test_data, test_label)

        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        #val
        avg_mse = 0
        avg_nmse = 0
        x1sum = 0
        y1sum = 0
        net_g.eval()
        criterionMSE = nn.MSELoss().to(device)
        with torch.no_grad():
            for batch in val_loader:
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input)
                x1 = criterionMSE(prediction, target)
                y1 = criterionMSE(target, torch.zeros_like(target))
                x1sum += x1.item();
                y1sum += y1.item();
                avg_mse += x1.item() / len(val_loader)
        avg_nmse = 10 * log10(x1sum / y1sum)
        print("===> valAvg. MSE: {:.4f}".format(avg_mse))
        print("===> valAvg. NMSE: {:.4f} dB".format(avg_nmse))
        y_val_mse.append(avg_mse)
        y_val_loss.append(avg_nmse)


        # test
        avg_mse = 0
        avg_nmse = 0
        x1sum = 0
        y1sum = 0
        net_g.eval()
        criterionMSE = nn.MSELoss().to(device)
        with torch.no_grad():
            for batch in test_loader:
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input)
                x1 = criterionMSE(prediction, target)
                y1 = criterionMSE(target, torch.zeros_like(target))
                x1sum += x1.item();
                y1sum += y1.item();
                avg_mse += x1.item() / len(test_loader)
        avg_nmse = 10 * log10(x1sum / y1sum)
        print("===> testAvg. MSE: {:.4f}".format(avg_mse))
        print("===> testAvg. NMSE: {:.4f} dB".format(avg_nmse))
        y_test_mse.append(avg_mse)
        y_test_loss.append(avg_nmse)

    return y_val_mse,y_val_nmse,y_test_mse,y_test_nmse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='channelestimation_test')
    parser.add_argument('--cuda', action='store_false', help='use cuda')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    path = './SISOCDL/test_onedmrs/test_for_figure/video/'
    path_val = path + 'val_30km_B_300_slot10'
    path_test = path + 'test_30km_B_300_slot10'
    # path = './SISOCDL/test_onedmrs/test_for_figure/video/'
    # path_val = path + 'val_100km_B_300'
    # path_test = path + 'test_100km_B_300'
    # path = './SISOCDL/test_onedmrs/test_for_figure/video/'
    # path_val = path + 'val_30km_C_300'
    # path_test = path + 'test_30km_C_300'

    # slot = 3
    # model_path = "./checkpoint/video_gan_2slot_onedmrs_14_3_slot/best_val_netG_model_epoch_810.pth"
    # slot = 5
    # model_path = "./checkpoint/video_gan_2slot_onedmrs_14_5_slot/best_val_netG_model_epoch_810.pth"
    slot = 7
    model_path = "./checkpoint/video_gan_2slot_onedmrs_14_7_slot/best_val_netG_model_epoch_810.pth"

    T: int =10
    batch_train_and_val = 24000
    batch_size = 64



    y_val,y_val_loss,y_test,y_test_loss = get_dataset(path_val, path_test,model_path,device,batch_size,slot=slot,T=T)

    x = floatrange(0, 20, 1.25)
    l1, = plt.plot(x, y_val_loss, marker='.', color='#1661ab', linestyle='dotted')
    l2, = plt.plot(x, y_test_loss, marker='.', color='xkcd:orange', linestyle='dotted')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-1, 21)

    plt.legend(handles=[l1, l2],
               labels=['inner', 'outside'], loc='best')

    plt.show()
    print(y_val)
    print(y_val_loss)
    print(y_test)
    print(y_test_loss)
    filename = open('video_onedmrs_val_nmse.txt', 'a')
    for value in y_val_loss:
        filename.write(str(value)+',')
    filename.write('\n')
    filename.close()
    filename = open('video_onedmrs_test_nmse.txt', 'a')
    for value in y_test_loss:
        filename.write(str(value)+',')
    filename.write('\n')
    filename.close()

import argparse
from math import log10
import h5py
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import get_allfile, h5_read_data, floatrange

def h5_read_data(filename, dataset):
    with h5py.File(filename, 'r') as f:
        dset = f[dataset]
        return dset[:]

def get_dataset(path_val, path_test,model_path,device,batch_size,slot=2,T=5):  # 获取所有文件

    with torch.no_grad():
        net_g = torch.load(model_path,map_location='cuda:0').to(device)
    y_val_mse = []
    y_val_nmse =[]
    y_test_mse = []
    y_test_nmse =[]

    all_file_val = get_allfile(path_val)  # tickets要获取文件夹名
    all_file_test = get_allfile(path_test)  # tickets要获取文件夹名
    all_file_val.sort()
    all_file_test.sort()
    print(all_file_val)
    print(all_file_test)
    # gt
    perfect = []
    video = []
    perfect_test = []
    video_test = []
    for i_f in range(len(all_file_val)):
        print(i_f)
        val_data = h5_read_data(all_file_val[i_f], 'ls')
        val_label = h5_read_data(all_file_val[i_f], 'gt')
        test_data = h5_read_data(all_file_test[i_f], 'ls')
        test_label = h5_read_data(all_file_test[i_f], 'gt')

        val_data = torch.from_numpy(val_data[:, :, T - slot:T:1, :, :])
        val_label = torch.from_numpy(val_label)
        test_data = torch.from_numpy(test_data[:, :, T - slot:T:1, :, :])
        test_label = torch.from_numpy(test_label)

        val_dataset = TensorDataset(val_data, val_label)
        test_dataset = TensorDataset(test_data, test_label)

        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        #val
        avg_mse = 0
        avg_nmse = 0
        x1sum = 0
        y1sum = 0
        net_g.eval()
        criterionMSE = nn.MSELoss().to(device)
        with torch.no_grad():
            for batch in val_loader:
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input)
                x1 = criterionMSE(prediction, target)
                y1 = criterionMSE(target, torch.zeros_like(target))
                x1sum += x1.item();
                y1sum += y1.item();
                avg_mse += x1.item() / len(val_loader)
        avg_nmse = 10 * log10(x1sum / y1sum)
        print("===> valAvg. MSE: {:.4f}".format(avg_mse))
        print("===> valAvg. NMSE: {:.4f} dB".format(avg_nmse))
        y_val_mse.append(avg_mse)
        y_val_loss.append(avg_nmse)


        # test
        avg_mse = 0
        avg_nmse = 0
        x1sum = 0
        y1sum = 0
        net_g.eval()
        criterionMSE = nn.MSELoss().to(device)
        with torch.no_grad():
            for batch in test_loader:
                input, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input)
                x1 = criterionMSE(prediction, target)
                y1 = criterionMSE(target, torch.zeros_like(target))
                x1sum += x1.item();
                y1sum += y1.item();
                avg_mse += x1.item() / len(test_loader)
        avg_nmse = 10 * log10(x1sum / y1sum)
        print("===> testAvg. MSE: {:.4f}".format(avg_mse))
        print("===> testAvg. NMSE: {:.4f} dB".format(avg_nmse))
        y_test_mse.append(avg_mse)
        y_test_loss.append(avg_nmse)

    return y_val_mse,y_val_nmse,y_test_mse,y_test_nmse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='channelestimation_test')
    parser.add_argument('--cuda', action='store_false', help='use cuda')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    path = './SISOCDL/test_onedmrs/test_for_figure/video/'
    path_val = path + 'val_30km_B_300_slot10'
    path_test = path + 'test_30km_B_300_slot10'
    # path = './SISOCDL/test_onedmrs/test_for_figure/video/'
    # path_val = path + 'val_100km_B_300'
    # path_test = path + 'test_100km_B_300'
    # path = './SISOCDL/test_onedmrs/test_for_figure/video/'
    # path_val = path + 'val_30km_C_300'
    # path_test = path + 'test_30km_C_300'

    # slot = 3
    # model_path = "./checkpoint/video_gan_2slot_onedmrs_14_3_slot/best_val_netG_model_epoch_810.pth"
    # slot = 5
    # model_path = "./checkpoint/video_gan_2slot_onedmrs_14_5_slot/best_val_netG_model_epoch_810.pth"
    slot = 7
    model_path = "./checkpoint/video_gan_2slot_onedmrs_14_7_slot/best_val_netG_model_epoch_810.pth"

    T: int =10
    batch_train_and_val = 24000
    batch_size = 64



    y_val,y_val_loss,y_test,y_test_loss = get_dataset(path_val, path_test,model_path,device,batch_size,slot=slot,T=T)

    x = floatrange(0, 20, 1.25)
    l1, = plt.plot(x, y_val_loss, marker='.', color='#1661ab', linestyle='dotted')
    l2, = plt.plot(x, y_test_loss, marker='.', color='xkcd:orange', linestyle='dotted')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(-1, 21)

    plt.legend(handles=[l1, l2],
               labels=['inner', 'outside'], loc='best')

    plt.show()
    print(y_val)
    print(y_val_loss)
    print(y_test)
    print(y_test_loss)
    filename = open('video_onedmrs_val_nmse.txt', 'a')
    for value in y_val_loss:
        filename.write(str(value)+',')
    filename.write('\n')
    filename.close()
    filename = open('video_onedmrs_test_nmse.txt', 'a')
    for value in y_test_loss:
        filename.write(str(value)+',')
    filename.write('\n')
    filename.close()
