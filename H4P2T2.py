#!/usr/bin/env python
# coding: utf-8

# ### Build a DNN for ECG Signal Classification (5 classes)  - Pytorch
# This is the implementation of an MLP for classifying the ECG signals. <br>
# Your task is to design new DNNs for ECG signal classification <br>
# You can use this file as a template

# In[83]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
import torch.optim as optim


# ## The Neural Network: MLP  (Replace this with your network and rename the file)

# In[84]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32,kernel_size=5, stride=2, padding=1)
        self.pool1 = nn.AvgPool1d(kernel_size = 2, padding = 0)
        
        self.conv2 = nn.Conv1d(32,32,5,2,1)
        self.pool2 = nn.AvgPool1d(kernel_size = 2)
        
        self.norm3 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,32,5,2,2)
        self.pool3 = nn.AvgPool1d(kernel_size = 2)
        
        self.norm4 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32,32,5,2,1)
        
        self.norm5 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(32,32,5,1,0)
        
        self.fc1 = nn.Linear(in_features=224, out_features=64)
        self.fc2 = nn.Linear(64, 5)
        
        
    def forward(self, x):
        x = x.view(x.size(0),1,x.size(1))
        
        y1 = nnF.relu(self.conv1(x))
        #print("y1", y1.shape)
        x1 = self.pool1(x)
        #print("x1", x1.shape)
        y1 = y1 + x1
        
        y2 = nnF.relu(self.conv2(y1))
        #print("y2", y2.shape)
        x2 = self.pool2(y1)
        #print("x2", x2.shape)
        y2 = y2 + x2
        
        y3 = self.norm3(y2)
        y3 = nnF.relu(self.conv3(y3))
        #print("y3", y3.shape)
        x3 = self.pool3(y2)
        #print("x3", x3.shape)
        y3 = y3 + x3
        
        y4 = self.norm4(y3)
        #print("y4", y4.shape)
        y4 = nnF.relu(self.conv4(y4))
        x4 = self.pool3(y3)
        #print("x4", x4.shape)
        y4 = y4 + x4
        
        y5 = self.norm5(y4)
        y5 = nnF.relu(self.conv5(y5))
        #print("y5", y5.shape)
        
        
        x = y5.view(y5.size(0),-1)
    
        
        x = nnF.relu(self.fc1(x))
        
        z = nnF.relu(self.fc2(x))
        #y=nnF.softmax(z, dim=1)
        return z


# In[85]:


#debug your network here
#make sure it works for one single input sample
model=Net()

x=torch.rand(10,187) # if network is MLP
#x=torch.rand(10,1,187) #if network is CNN
z=model(x)


# In[86]:


def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)


# In[87]:


def cal_accuracy(confusion):
    #input: confusion is the confusion matrix
    #output: acc is the standard classification accuracy
    M=confusion.copy().astype('float32')
    acc = M.diagonal().sum()/M.sum()    
    sens=np.zeros(M.shape[0])
    prec=np.zeros(M.shape[0]) 
    for n in range(0, M.shape[0]):
        TP=M[n,n]
        FN=np.sum(M[n,:])-TP
        FP=np.sum(M[:,n])-TP
        sens[n]=TP/(TP+FN)
        prec[n]=TP/(TP+FP)       
    return acc, sens, prec


# ## The function to train the model

# In[88]:


def train(model, device, optimizer, dataloader, epoch):    
    model.train()#set model to training mode
    loss_train=0
    acc_train =0 
    sample_count=0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()#clear grad of each parameter
        Z = model(X)#forward pass
        loss = nnF.cross_entropy(Z, Y)
        loss.backward()#backward pass
        optimizer.step()#update parameters
        loss_train+=loss.item()
        #do not need softmax
        Yp = Z.data.max(dim=1)[1]  # get the index of the max               
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 100. * batch_idx / len(dataloader), loss.item()))
    loss_train/=len(dataloader)
    #due to upsampling, len(dataloader.dataset) != sample_count
    #acc_train/=len(dataloader.dataset) 
    acc_train/=sample_count    
    return loss_train, acc_train


# ## The Function to test the model

# In[89]:


def test(model, device, dataloader):
    model.eval()#set model to evaluation mode
    acc_test =0
    confusion=np.zeros((5,5))
    with torch.no_grad(): # tell Pytorch not to build graph in the with section
        for batch_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            Z = model(X)#forward pass
            #do not need softmax
            Yp = Z.data.max(dim=1)[1]  # get the index of the max 
            acc_test+= torch.sum(Yp==Y).item()
            for i in range(0, 5):
                for j in range(0, 5):
                    confusion[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
    acc, sens, prec=cal_accuracy(confusion)
    return acc, (confusion, sens, prec)


# ## Load data and create dataloaders

# In[90]:


class MyDataset(torch_dataset):
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        #return the number of data points
        return self.X.shape[0]
    def __getitem__(self, idx):        
        #we can use DatasetName[idx] to get a data point (x,y) with index idx
        x=torch.tensor(self.X[idx], dtype=torch.float32)
        y=torch.tensor(self.Y[idx], dtype=torch.int64)
        #x=x.reshape(1,-1) if network is CNN
        return x, y


# In[91]:


import pandas as pd
X=pd.read_csv('ECG_dataX.csv')
Y=pd.read_csv('ECG_dataY.csv')


# In[92]:


#convert dataframe to numpy array
X=X.values
X.shape


# In[93]:


#convert dataframe to numpy array
Y=Y.values
Y.shape


# In[94]:


#reshape Y into a 1D array
Y=Y.reshape(-1)
Y.shape


# In[95]:


plt.hist(Y)


# In[96]:


fs=125  # sampling frequency
Ts=1/fs # sampling interval
N=187 # the number of timepoints
Duration=N*Ts # duration of a signal
t=np.linspace(0, Duration-Ts, N) # array of timepoints
fig, ax = plt.subplots(5,1,constrained_layout=True, figsize=(10,10))
for c in range(0, 5):   
    for n in range(0, 3):
        idx=np.random.randint(0,10)
        ax[c].plot(t, X[Y==c][idx])        
        ax[c].set_xlabel('time t [seconds]', fontsize=16)
        ax[c].set_ylabel('amplitude', fontsize=16)
    ax[c].set_title('class '+str(c), fontsize=16)


# In[97]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)


# In[98]:


dataset_train=MyDataset(X_train, Y_train)
dataset_val=MyDataset(X_val, Y_val)
dataset_test=MyDataset(X_test, Y_test)


# In[99]:


loader_train = torch_dataloader(dataset_train, batch_size=128, shuffle=True, num_workers=0)
loader_val = torch_dataloader(dataset_val, batch_size=128, shuffle=False, num_workers=0) 
loader_test = torch_dataloader(dataset_test, batch_size=128, shuffle=False, num_workers=0) 


# ## Create a model, and start the traning-validation-testing process

# In[100]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=Net()
model.to(device)
#the code below may cause error if BatchNorm is used, you may delete those without affecting the rest of the file
#x=dataset_train[0][0]
#x=x.view(1,1,187).to(device) #change it to x=x.view(1,1,187).to(device) for CNN
#z=model(x)


# use stochastic gradient descent as the optimization method

# In[101]:


optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=1e-4)


# In[102]:


loss_train_list=[]
acc_train_list=[]
acc_val_list=[]
epoch_save=-1


# train/val/test over many epochs

# In[103]:


for epoch in range(epoch_save+1, 100): #change 100 to a larger number if necessary
    #-------- training --------------------------------
    loss_train, acc_train =train(model, device, optimizer, loader_train, epoch)    
    loss_train_list.append(loss_train)
    acc_train_list.append(acc_train)
    print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
    #-------- validation --------------------------------
    acc_val, other_val = test(model, device, loader_val)
    acc_val_list.append(acc_val)
    print('epoch', epoch, 'validation acc:', acc_val)
    #--------save model-------------------------
    result = (loss_train_list, acc_train_list, 
              acc_val_list, other_val)
    save_checkpoint('LUKECAPRIO_ECG_CNN_Pytorch_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
    epoch_save=epoch


# In[23]:


fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].set_title('loss v.s. epoch',fontsize=16)
ax[0].plot(loss_train_list, '-b', label='training loss')
ax[0].set_xlabel('epoch',fontsize=16)
ax[0].legend(fontsize=16)
ax[0].grid(True)
ax[1].set_title('accuracy v.s. epoch',fontsize=16)
ax[1].plot(acc_train_list, '-b', label='training accuracy')
ax[1].plot(acc_val_list, '-g', label='validation accuracy')
ax[1].set_xlabel('epoch',fontsize=16)
ax[1].legend(fontsize=16)
ax[1].grid(True)


# load the best model

# In[24]:


best_epoch=np.argmax(acc_val_list)
best_epoch


# In[25]:


checkpoint=torch.load('LUKECAPRIO_ECG_CNN_Pytorch_epoch'+str(best_epoch)+'.pt')
model=Net()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device);
model.eval();


# In[26]:


#third try
acc, (confusion, sens, prec) = test(model, device, loader_test)
print('Accuracy (average)', acc)
print('Sensitivity', sens)
print('Precision', prec)
print('Confusion \n', confusion)


# In[67]:


acc, (confusion, sens, prec) = test(model, device, loader_test)
print('Accuracy (average)', acc)
print('Sensitivity', sens)
print('Precision', prec)
print('Confusion \n', confusion)


# In[25]:


acc, (confusion, sens, prec) = test(model, device, loader_test)
print('Accuracy (average)', acc)
print('Sensitivity', sens)
print('Precision', prec)
print('Confusion \n', confusion)


# In[ ]:




