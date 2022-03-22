# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:31:52 2021

@author: OuYang
"""

# 1 导入库
import torch.nn as nn
import torch.nn.functional as F

# RCNN的CNN结构
class CNN(nn.Module):
    def __init__(self,L):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*int(L/4)*int(L/4),1) #After two maxpooling 
    
    def forward(self,x):
        # 防止报类型错误
        x = x.float() # 避免类型不同报错
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

# M-RCNN的CNN结构  
class CNN1(nn.Module):
    def __init__(self,L):
        super(CNN1,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        #self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*int(L/4)*int(L/4),1) #After two maxpooling 
    
    def forward(self,x):
        # 防止报类型错误
        x = x.float() # 避免类型不同报错
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        #x = F.relu(self.conv3(x))
        #x = self.MaxPool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x

# 双通道M-RCNN框架
class CNN2(nn.Module):
    def __init__(self,L):
        super(CNN2,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        #self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*int(L/4)*int(L/4),1) #After two maxpooling 
    
    def forward(self,x):
        # 防止报类型错误
        x = x.float() # 避免类型不同报错
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        #x = F.relu(self.conv3(x))
        #x = self.MaxPool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x