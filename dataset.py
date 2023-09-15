import glob
import os
import h5py
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, Subset
import tqdm

from  progressbar import ProgressBar

class TrainDataset(Dataset):
    def __init__(self, img,date ,label):

        self.img = img
        self.label = label
        self.date = date

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        frames = self.img[idx]
        date_data = self.date[idx]
        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)  # THWC -> CTHW
        return frames, date_data ,label

class TestDataset(Dataset):
    def __init__(
        self,
        test_data,
        date,
        label,
        transform
    ):
        self.data  = test_data
        self.label = label
        self.transform = transform
        self.date = date

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.label[idx]
        frames = self.data[idx]
        date_data = self.date[idx]

        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)

        if self.transform:
            frames = self.transform(frames)

        return frames,date_data, label       # dummy label

class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        frames, date_data,label = super().__getitem__(idx)
        if self.transform:
            frames = self.transform(frames)
        return frames,date_data, label

def MSE(pred, true):

    if len(pred) != len(true):
        raise ValueError("pred and true aren't same length")

    load = np.zeros(len(pred))

    for i in range(len(pred)):
        load[i] = (pred[i] - true[i])**2

    MSE = sum(load)/len(load)

    return MSE

def test(net, testLoader,device):
    net.eval()
    totalLoss = 0
    with torch.no_grad():
        for data,date, label in testLoader:
            data = data.to(device,dtype=torch.float)
            date = date.to(device,dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            data = data.float()

            output = net(data,date)

            loss = MSE(output, label)

            totalLoss += loss.item()

    return totalLoss

def pred(net , test_loader, device ):

    net.eval()
    pred = np.zeros((len(test_loader.dataset), 1))
    i = 0

    with torch.no_grad():
        for data,date, _ in test_loader:
            data = data.to(device, dtype=torch.float)
            date = date.to(device,dtype=torch.float)

            output = net(data,date)

            test_label = output.cpu().data.numpy()

            pred[i:i+len(test_label)] = test_label
            i += len(test_label)

    return pred

def train(model,epochs,trigger_times,optimizer,criterion,train_loader,val_loader,device,scheduler,time_stamp,train_writer):

    temp_loss = float("inf")
    loss_his = []
    val_his = []
    count = 0
    current_best_loss = float("inf")

    progress_bar = ProgressBar(epochs)

    for epoch in range(1, epochs+1):

        progress_bar = ProgressBar(len(train_loader))
        model.train()

        total_loss = 0

        for idx, (data_x,date, label) in enumerate(train_loader):
            progress_bar.update()
            data_x = data_x.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            date = date.to(device,dtype=torch.float)

            label = label.reshape(len(label),1)


            optimizer.zero_grad()

            output = model(data_x,date)
            loss = criterion(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            loss_his.append(loss.item())


        current_val_loss = test(model, val_loader,device)
        val_his.append(current_val_loss / len(val_loader))
        scheduler.step(current_val_loss)
        print(f"| epoch : [{epoch}/{epochs}] | Train Loss: {total_loss / len(train_loader):.4f} |  Val Loss: {current_val_loss / len(val_loader):.4f} |")

        train_writer.add_scalar('myscalar' ,current_val_loss / len(val_loader),epoch)

        if current_val_loss >= temp_loss:

            count += 1

            if count >= trigger_times:
                print(f"Early stopping triggered after epoch {epoch}")
                break

        else:
            count = 0

        temp_loss = current_val_loss

        if current_best_loss >= temp_loss:

            current_best_loss = temp_loss

            bestModel = model

            print("Saving best model")

            torch.save(bestModel.state_dict(), "model_save/epoch"+str(epoch+1)+"_"+str(current_best_loss)+".pt")

            bset_model_path = "model_save/epoch"+str(epoch+1)+"_"+str(current_best_loss)+".pt"

    print("Training complete!")

    train_writer.add_graph(model,(data_x))
    train_writer.close()


    return bestModel ,bset_model_path

