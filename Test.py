# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:54:48 2021

@author: OuYang
"""
# 1 导入库
import numpy as np
import torch
import Embeddings
import Utils
import Models
import networkx as nx
from scipy import stats

# RCNN转换torch
def to_torch1(data,L):
    """把数据转化为torch格式
    Parameters:
        data:main函数生成的数据（字典格式）
        L:目标节点+邻居节点的数量
    return:
        torch_data:torch格式
    """
    torch_data = torch.empty(len(data),1,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_data[inx,:,:] = torch.from_numpy(matrix)
    return torch_data

# M-RCNN转换torch
def to_torch2(data,L):
    """把数据转化为torch格式
    Parameters:
        data:main函数生成的数据（字典格式）
        L:目标节点+邻居节点的数量
    return:
        torch_data:torch格式
    """
    torch_data = torch.empty(len(data),3,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_data[inx,:,:] = matrix
    return torch_data

# M-RCNN转换torch
def to_torch3(data,L):
    """把数据转化为torch格式
    Parameters:
        data:main函数生成的数据（字典格式）
        L:目标节点+邻居节点的数量
    return:
        torch_data:torch格式
    """
    torch_data = torch.empty(len(data),2,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_data[inx,:,:] = matrix
    return torch_data

def nodesRank(rank):
    SR = sorted(rank)
    re = []
    for i in SR:
        re.append(rank.index(i))
    return re

def order(result):
    """传入一个字典，根据值对各个键进行升序排序
    Parameters:
        result:一个字典，其中键是节点名，值是节点的得分
        
    return:
        order_list:各个节点的排名情况
    
    """
    n = len(result)-1
    for inx,(k,v) in enumerate(sorted(result.items(),key=lambda x:x[1],reverse=False)):
        result[k] = n-inx
    return list(result.values())

def compare_tau(G,L1,L2,sir_list,community,RCNN,MRCNN):
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        dc:度中心性
        bc:介数中心性
        ks:k-shell
        sir_list:不同beta下的SIR模拟结果
        model:训练好的模型
        p:选择比较节点的比例
    return:
        tau_list:在不同beta情况下的tau值
    """
    dc = dict(nx.degree_centrality(G))
    ks = dict(nx.core_number(G))
    bc = dict(nx.betweenness_centrality(G))
    nd = Utils.neighbor_degree(G)
    vc = Utils.VC(G,community)
    nodes = list(G.nodes())
    rcnn_data = to_torch1(Embeddings.main(G,L1),L1)
    mrcnn_data = to_torch2(Embeddings.main1(G,L2,community),L2)
    rcnn_pred = [i for i,j in sorted(dict(zip(nodes,RCNN(rcnn_data))).items(),key=lambda x:x[1],reverse=True)] # 获得RCNN预测的节点重要性，降序排序
    rcnn_rank = np.array(nodesRank(rcnn_pred),dtype=float)
    
    mrcnn_pred = [i for i,j in sorted(dict(zip(nodes,MRCNN(mrcnn_data))).items(),key=lambda x:x[1],reverse=True)] # 获得RCNN预测的节点重要性，降序排序
    mrcnn_rank = np.array(nodesRank(mrcnn_pred),dtype=float)
    
    dc_rank = np.array(nodesRank([i for i,j in sorted(dc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
    ks_rank = np.array(nodesRank([i for i,j in sorted(ks.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
    bc_rank = np.array(nodesRank([i for i,j in sorted(bc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
    nd_rank = np.array(nodesRank([i for i,j in sorted(nd.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
    vc_rank = np.array(nodesRank([i for i,j in sorted(vc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
    
    RCNN_tau_list = []
    MRCNN_tau_list = []
    dc_tau_list = []
    ks_tau_list = []
    nd_tau_list = []
    bc_tau_list = []
    vc_tau_list = []
    
    for sir in sir_list:
        sir_sort = [i for i,j in sorted(sir.items(),key=lambda x:x[1],reverse=True)]
        sir_rank = np.array(nodesRank(sir_sort),dtype=float)
        tau1,_ = stats.kendalltau(rcnn_rank,sir_rank)
        tau2,_ = stats.kendalltau(mrcnn_rank,sir_rank)
        tau3,_ = stats.kendalltau(dc_rank,sir_rank)
        tau4,_ = stats.kendalltau(ks_rank,sir_rank)
        tau5,_ = stats.kendalltau(nd_rank,sir_rank)
        tau6,_ = stats.kendalltau(bc_rank,sir_rank)
        tau7,_ = stats.kendalltau(vc_rank,sir_rank)
        
        RCNN_tau_list.append(tau1)
        MRCNN_tau_list.append(tau2)
        dc_tau_list.append(tau3)
        ks_tau_list.append(tau4)
        nd_tau_list.append(tau5)
        bc_tau_list.append(tau6)
        vc_tau_list.append(tau7)
    return RCNN_tau_list,MRCNN_tau_list,dc_tau_list,ks_tau_list,nd_tau_list,bc_tau_list,vc_tau_list

def compare_Train_Test_tau(G,community,L_list,sir,model_list,p=0.1):
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        dc:度中心性
        bc:介数中心性
        ks:k-shell
        sir_list:不同beta下的SIR模拟结果
        model:训练好的模型
        p:选择比较节点的比例
    return:
        tau_list:在不同beta情况下的tau值
    """
    num_nodes = int(p*G.number_of_nodes()) # 要对比的节点数量
    nodes = list(G.nodes())
    tau_list = []
    sir_sort = [i for i,j in sorted(sir.items(),key=lambda x:x[1],reverse=True)]
    sir_rank = np.array(nodesRank(sir_sort),dtype=float)
    for i,L in enumerate(L_list):
        data = to_torch2(Embeddings.main1(G,L,community),L)
        model = model_list[i]
        pred = [i for i,j in sorted(dict(zip(nodes,model(data))).items(),key=lambda x:x[1],reverse=True)] # 获得RCNN预测的节点重要性，降序排序
        rank = np.array(nodesRank(pred),dtype=float)
        tau,_ = stats.kendalltau(rank,sir_rank)
        tau_list.append(tau)
    return tau_list

def Taus(G,L_list,label,models):
    taus = []
    for i in range(6):
        model_list=models[i]
        r_tau = compare_Train_Test_tau(G,L_list,label,model_list)
        taus.append(r_tau)
    return taus

def compare_tau1(G,L,sir_list,community,MRCNN):
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        dc:度中心性
        bc:介数中心性
        ks:k-shell
        sir_list:不同beta下的SIR模拟结果
        model:训练好的模型
        p:选择比较节点的比例
    return:
        tau_list:在不同beta情况下的tau值
    """
    num_nodes = int(p*G.number_of_nodes()) # 要对比的节点数量
    nodes = list(G.nodes())
    mrcnn_data = to_torch2(Embeddings.main1(G,L,community),L)
    mrcnn_pred = [i for i,j in sorted(dict(zip(nodes,MRCNN(mrcnn_data))).items(),key=lambda x:x[1],reverse=True)]
    mrcnn_rank = np.array(nodesRank(mrcnn_pred),dtype=float)
    
    MRCNN_tau_list = [] 
    for sir in sir_list:
        sir_sort = [i for i,j in sorted(sir.items(),key=lambda x:x[1],reverse=True)]
        sir_rank = np.array(nodesRank(sir_sort),dtype=float)
        tau1,_ = stats.kendalltau(mrcnn_rank,sir_rank)
        MRCNN_tau_list.append(tau1)
    return MRCNN_tau_list,np.mean(MRCNN_tau_list)

def calculate_improve(G,L1,L2,SIR_list,community,RCNN,MRCNN,method):
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        dc:度中心性
        bc:介数中心性
        ks:k-shell
        sir_list:不同beta下的SIR模拟结果
        model:训练好的模型
        p:选择比较节点的比例
    return:
        tau_list:在不同beta情况下的tau值
    """
    nodes = list(G.nodes())
    rcnn_data = to_torch1(Embeddings.main(G,L1),L1)
    mrcnn_data = to_torch3(Embeddings.main2(G,L2,community,method=method),L2)
    rcnn_pred = [i for i,j in sorted(dict(zip(nodes,RCNN(rcnn_data))).items(),key=lambda x:x[1],reverse=True)] # 获得RCNN预测的节点重要性，降序排序
    rcnn_rank = np.array(nodesRank(rcnn_pred),dtype=float)
    
    mrcnn_pred = [i for i,j in sorted(dict(zip(nodes,MRCNN(mrcnn_data))).items(),key=lambda x:x[1],reverse=True)] # 获得RCNN预测的节点重要性，降序排序
    mrcnn_rank = np.array(nodesRank(mrcnn_pred),dtype=float)
    
    improve_rates = []
    
    for sir in sir_list:
        sir_sort = [i for i,j in sorted(sir.items(),key=lambda x:x[1],reverse=True)]
        sir_rank = np.array(nodesRank(sir_sort),dtype=float)
        tau1,_ = stats.kendalltau(rcnn_rank,sir_rank)
        tau2,_ = stats.kendalltau(mrcnn_rank,sir_rank)
        
        improve_rates.append((tau2-tau1)/tau1)

    return improve_rates

def calculate_similarity(G,L1,L2,SIR_dic,community,RCNN,MRCNN):
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        dc:度中心性
        bc:介数中心性
        ks:k-shell
        sir_list:不同beta下的SIR模拟结果
        model:训练好的模型
        p:选择比较节点的比例
    return:
        tau_list:在不同beta情况下的tau值
    """
    nodes = list(G.nodes())
    rcnn_data = to_torch1(Embeddings.main(G,L1),L1)
    mrcnn_data = to_torch2(Embeddings.main1(G,L2,community),L2)
    rcnn_pred = dict(zip(nodes,RCNN(rcnn_data))) # 获得RCNN预测的节点重要性，降序排序
    mrcnn_pred = dict(zip(nodes,MRCNN(mrcnn_data))) # 获得RCNN预测的节点重要性，降序排序
    RCNN_tau_list = []
    MRCNN_tau_list = []
    SIR_list = []
    for node in nodes:
        RCNN_tau_list.append(rcnn_pred[node])
        MRCNN_tau_list.append(mrcnn_pred[node])
        SIR_list.append(SIR_dic[node])
    result_pd = pd.DataFrame({'SIR':SIR_list,'RCNN':RCNN_tau_list,'MRCNN':MRCNN})
    return result_pd


def cal_input_time(G,label,L_list,batch_size,num_epochs,lr):
    input_time = {}
    for L in L_list:
        start_time = time.time()
        train_data = Embeddings.main(G,L)
        end_time = time.time()
        input_time[str(L)] = (end_time-start_time)
    return input_time

def cal_input_time2(G,label,L_list,batch_size,num_epochs,lr):
    input_time = {}
    for L in L_list:
        start_time = time.time()
        _,com,_ = Utils.Louvain(G)
        train_data = Embeddings.main1(G,L,com)
        end_time = time.time()
        input_time[str(L)] = (end_time-start_time)
    return input_time


def cal_training_time(G,label,L_list,batch_size,num_epochs,lr):
    training_time = {}
    for L in L_list:
        start_time = time.time()
        train_data = Embeddings.main(G,L)
        data_loader = Utils.Get_DataLoader(train_data,label,batch_size,L)
        rcnn= Models.CNN(L)
        RCNN,RCNN_loss = Utils.train_model(data_loader,rcnn,num_epochs,lr,L)
        end_time = time.time()
        training_time[str(L)] = (end_time-start_time)
    return training_time

def cal_training_time2(G,label,L_list,batch_size,num_epochs,lr):
    training_time = {}
    for L in L_list:
        start_time = time.time()
        train_data = Embeddings.main1(G,L)
        data_loader = Utils.Get_DataLoader1(train_data,label,batch_size,L)
        mrcnn= Models.CNN1(L)
        MRCNN,MRCNN_loss = Utils.train_model(data_loader,mrcnn,num_epochs,lr,L)
        end_time = time.time()
        training_time[str(L)] = (end_time-start_time)
    return training_time

    
      