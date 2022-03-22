# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:17:03 2021

@author: OuYang
"""
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import community
import Test
import Embeddings
import random
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
sns.set_style('ticks')
os.chdir('D:/研究生文献/文献复现工作/RCNN整理后代码+源码+数据/')


def generate_subgraph(G,node_list):
    """根据目标节点获取邻接矩阵
    Parameters:
        G:目标网络
        node_list:包含目标节点和固定数量邻居节点的列表
    return:
        G_sub: 子网络
        A: 对应的邻居网络
    """
    G_sub = nx.Graph()
    L = len(node_list) 
    encode = dict(zip(node_list,list(range(L)))) # 对节点进行重新编号
    subgraph = nx.subgraph(G,node_list) # 提取子图
    subgraph_edges = list(subgraph.edges()) # 获取子图的边列表
    new_subgraph_edges = []
    for i,j in subgraph_edges:
        new_subgraph_edges.append((encode[i],encode[j]))
    G_sub.add_edges_from(new_subgraph_edges)
    A = np.zeros([L,L])
    for i in range(L):
        for j in range(L):
            if G_sub.has_edge(i,j) and (i!=j):
                A[i,j]=1
    return G_sub,A

def transform(A,degree_list):
    """按规则进行转换
    Parameters:
        A: 邻接矩阵
        degree_list:所选节点对应的度值
    return:
        B:单通道的嵌入矩阵
    """
    B = A
    B[0,1:] = A[0,1:]*(np.array(degree_list)[1:])
    B[1:,0] = A[1:,0]*(np.array(degree_list)[1:])
    for i in range(len(degree_list)):
        B[i,i]=degree_list[i]
    return B

def transform1(A,degree_list,com_list,shell_list):
    """按规则进行转换
    Parameters:
        A: 邻接矩阵
        degree_list:所选节点对应的度值
        com_list:所选节点对应的所连社团个数
        shell_list:所选节点对应的k核值
    return:
        B:3通道的嵌入矩阵
    """
    B1 = A.copy()
    B2 = A.copy()
    B3 = A.copy()
    B1[0,1:] = B1[0,1:]*(np.array(degree_list)[1:])
    B1[1:,0] = B1[1:,0]*(np.array(degree_list)[1:])

    B2[0,1:] = B2[0,1:]*(np.array(com_list)[1:])
    B2[1:,0] = B2[1:,0]*(np.array(com_list)[1:])
    
    B3[0,1:] = B3[0,1:]*(np.array(shell_list)[1:])
    B3[1:,0] = B3[1:,0]*(np.array(shell_list)[1:])

    for i in range(len(degree_list)):
        B1[i,i]=degree_list[i]
        B2[i,i]=com_list[i]
        B3[i,i]=shell_list[i]
    
    B = torch.zeros(3,A.shape[0],A.shape[0])
    B[0,:,:]= torch.from_numpy(B1).float()
    B[1,:,:]= torch.from_numpy(B2).float()
    B[2,:,:]= torch.from_numpy(B3).float()
    return B

def transform2(A,degree_list,com_list):
    """按规则进行转换
    Parameters:
        A: 邻接矩阵
        degree_list:所选节点对应的度值
        com_list:所选节点对应的所连社团个数
        shell_list:所选节点对应的k核值
    return:
        B:3通道的嵌入矩阵
    """
    B1 = A.copy()
    B2 = A.copy()
    B1[0,1:] = B1[0,1:]*(np.array(degree_list)[1:])
    B1[1:,0] = B1[1:,0]*(np.array(degree_list)[1:])

    B2[0,1:] = B2[0,1:]*(np.array(com_list)[1:])
    B2[1:,0] = B2[1:,0]*(np.array(com_list)[1:])
    

    for i in range(len(degree_list)):
        B1[i,i]=degree_list[i]
        B2[i,i]=com_list[i]
    
    B = torch.zeros(2,A.shape[0],A.shape[0])
    B[0,:,:]= torch.from_numpy(B1).float()
    B[1,:,:]= torch.from_numpy(B2).float()
    return B

def neighbor_degree(G):
    """邻居度：邻居节点的度之和"""
    nodes = list(G.nodes())
    degree = dict(G.degree())
    n_degree = {}
    for node in nodes:
        nd = 0
        neighbors = G.adj[node]
        for nei in neighbors:
            nd+=degree[nei]
        n_degree[node]=nd
    return n_degree

def Louvain(G):
    """使用Louvain算法进行社团划分"""
    def com_number(G,partition,community_dic):
        """获得每个节点所连社团个数与社团大小"""
        com_num = {}
        com_size = {}
        for node in G.nodes():
            com_size[node]=len(community_dic[partition[node]])
            com_set = set([partition[node]])
            for nei in list(G.adj[node]):
                if partition[nei] not in com_set:
                    com_set.add(partition[nei])
            com_num[node]=len(com_set)
        return com_num,com_size
    
    partition = community.best_partition(G)
    community_name = set(list(partition.values()))
    community_dic = {}
    for each in community_name:
        a = []
        for node in list(partition.keys()):
            if partition[node] == each:
                a.append(node)
        community_dic[each] = a
    com_num,com_size = com_number(G,partition,community_dic)
    return community_dic,com_num,com_size

def setup_seed(seed):
    """固定种子"""
    torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path,create_using=nx.Graph())
    return G

def load_sir_list(path):
    """
    读取不同beta情况模拟下的SIR的结果
    Parameters:
        path:存放SIR结果的根路径
    return:
        每个节点的SIR模拟结果
    """
    sir_list = []
    for i in range(10):
        sir = pd.read_csv(path+str(i)+'.csv')
        sir_list.append(dict(zip(np.array(sir['Node'],dtype=str),sir['SIR'])))
    return sir_list

# 单通道DataLoader
def Get_DataLoader(data,label,batch_size,L):
    """创建单DataLoader
    Parameters:
        data:数据集都为字典（键为节点，值为矩阵）
        label:数据集都为字典（键为节点，值为矩阵）
        batch_size:每次训练多少个
    return:
        Loader:DataLoader。
    
    """
# 首先把所有numpy转为torch
    torch_set = torch.empty(len(data),1,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_set[inx,:,:,:] = torch.from_numpy(matrix)

    sir_torch = torch.empty(len(label),1)
    for inx,v in enumerate(label.values()):
        sir_torch[inx,:] = v

    # 创建DataLoader
    deal_data = TensorDataset(torch_set,sir_torch)
    Loader = DataLoader(dataset=deal_data,batch_size=batch_size,shuffle=True)
    return Loader

# 三通道DataLoader
def Get_DataLoader1(data,label,batch_size,L):
    """创建三通道DataLoader
    Parameters:
        data:数据集都为字典（键为节点，值为矩阵）
        label:数据集都为字典（键为节点，值为矩阵）
        batch_size:每次训练多少个
    return:
        Loader:DataLoader。
    
    """
    # 首先把所有numpy转为torch
    torch_set = torch.empty(len(data),3,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_set[inx,:,:,:] = matrix

    sir_torch = torch.empty(len(label),1)
    for inx,v in enumerate(label.values()):
        sir_torch[inx,:] = v

    # 创建DataLoader
    deal_data = TensorDataset(torch_set,sir_torch)
    Loader = DataLoader(dataset=deal_data,batch_size=batch_size,shuffle=True)
    return Loader

def train_model(loader,model,num_epochs,lr,L,path=None):
    """训练模型
    Parameters:
        loader: pytorch dataloader
        num_epochs: 训练的轮数
        lr:学习率
        path:模型存放路径
    return:
        model:训练好的模型
        loss_list:不同轮数的loss变化
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss() # 损失函数
    optimizer = optim.Adam(model.parameters(),lr=lr) # 优化函数
    loss_list = [] # 存放loss的列表
    for epoch in tqdm(range(num_epochs)):
        for data,targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            pred = model(data)
            loss = criterion(pred,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(loss.data)
        if epoch % 100 == 0:
            print("Loss:{}".format(loss.data))
    # 绘制loss的变化
    plt.figure(figsize=(8,6),dpi=100)
    plt.xlabel('epochs',fontsize=14,fontweight='bold')
    plt.ylabel('loss',fontsize=14,fontweight='bold')
    plt.plot(np.arange(0,num_epochs,10),loss_list,marker='o',c='r',label='BA_1000_4_28')
    plt.legend()
    plt.show()
    if path:
        torch.save(model,path)
    return model,loss_list

def calculate(l1,l2):
    p = np.mean((np.array(l2)-np.array(l1))/np.array(l1))
    return p

def compare_L(G,label,community,L_list,batch_size=32):
    models = []
    for L in L_list:
        mrcnn_data= Embeddings.main1(G,L,community)
        mrcnn_loader = Utils.Get_DataLoader1(mrcnn_data,label,batch_size,L)
        mrcnn = Models.CNN1(L)
        MRCNN,MRCNN_loss = Utils.train_model(mrcnn_loader,mrcnn,500,0.001,L)
        models.append(MRCNN)
    return models

def normalization(data):
    data_norm = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
    return data_norm


def save_csv(rcnn,mrcnn,dc,ks,nd,bc,path):
    data = pd.DataFrame({'RCNN':rcnn,'MRCNN':mrcnn,'DC':dc,'K-core':ks,'ND':nd,'BC':bc})
    data.to_csv(path,index=False)
    
def extract_subgraph(G,node):
    candidates = set([node])
    for neighbor in list(G.adj[node]):
        candidates.add(neighbor)
    subnetwork = nx.subgraph(G, list(candidates))
    return subnetwork
        
def pred_position(G,L1,L2,rcnn,mrcnn,p=10):
    nodes = list(G.nodes())
    rcnn_data = Test.to_torch1(Embeddings.main(G,L1),L1)
    rcnn_data1,com = Embeddings.main1(G,L2)
    rcnn_data1 = Test.to_torch2(rcnn_data1,L2)
    rcnn_pred = [i for i,j in sorted(dict(zip(nodes,rcnn(rcnn_data))).items(),key=lambda x:x[1],reverse=True)][:p] # 获得RCNN预测的节点重要性，降序排序
    rcnn_pred1 = [i for i,j in sorted(dict(zip(nodes,mrcnn(rcnn_data1))).items(),key=lambda x:x[1],reverse=True)][:p] # 获得RCNN预测的节点重要性，降序排序
    subnetworks1 = []
    subnetworks2 = []
    for i in range(p):
        subnetworks1.append(extract_subgraph(G, rcnn_pred[i]))
        subnetworks2.append(extract_subgraph(G, rcnn_pred1[i]))
    return subnetworks1,subnetworks2,rcnn_pred,rcnn_pred1,com

def load_set(root):
    com = pd.read_csv(root+'com.csv')
    mrcnn = pd.read_csv(root+'mrcnn_toprank.csv')
    rcnn = pd.read_csv(root+'rcnn_toprank.csv')
    sir = pd.read_csv(root+'sir_toprank.csv')
    return com,mrcnn,rcnn,sir

def save_data(G,SIR_dict,RCNN,MRCNN,root,L1=28,L2=24):
    core = dict(nx.core_number(G))
    nd = neighbor_degree(G)
    RCNN_sub,MRCNN_sub,RCNN_nodes,MRCNN_nodes,com = pred_position(G,L1,L2,RCNN,MRCNN)
    sir_core = []
    sir_com = []
    sir_nd = []
    sir = []
    sir_rank = [i for i,j in sorted(SIR_dict.items(),key=lambda x:x[1],reverse=True)][:10]
    rcnn_core = []
    rcnn_sir = []
    mrcnn_sir = []
    mrcnn_core = []
    rcnn_nd = []
    mrcnn_nd = []
    rcnn_com = []
    mrcnn_com = []
    for i in range(len(RCNN_nodes)):
        rcnn_core.append(core[RCNN_nodes[i]])
        rcnn_com.append(com[RCNN_nodes[i]])
        rcnn_nd.append(nd[RCNN_nodes[i]])
        rcnn_sir.append(SIR_dict[RCNN_nodes[i]])
        
        mrcnn_core.append(core[MRCNN_nodes[i]])
        mrcnn_com.append(com[MRCNN_nodes[i]])
        mrcnn_nd.append(nd[MRCNN_nodes[i]])
        mrcnn_sir.append(SIR_dict[MRCNN_nodes[i]])
        
        sir_core.append(core[sir_rank[i]])
        sir_com.append(com[sir_rank[i]])
        sir_nd.append(nd[sir_rank[i]])
        sir.append(SIR_dict[sir_rank[i]])
        
    com_pd = pd.DataFrame({'Nodes':list(com.keys()),'Number of community':list(com.values())})
    rcnn_pd  = pd.DataFrame({'Nodes':RCNN_nodes,'K-core':rcnn_core,'Number of community':rcnn_com,'ND':rcnn_nd,'SIR':rcnn_sir})
    mrcnn_pd  = pd.DataFrame({'Nodes':MRCNN_nodes,'K-core':mrcnn_core,'Number of community':mrcnn_com,'ND':mrcnn_nd,'SIR':mrcnn_sir})
    sir_pd  = pd.DataFrame({'Nodes':sir_rank,'K-core':sir_core,'Number of community':sir_com,'ND':sir_nd,'SIR':sir})
    com_pd.to_csv(root+'com.csv',index=False)
    rcnn_pd.to_csv(root+'rcnn_toprank.csv',index=False)
    mrcnn_pd.to_csv(root+'mrcnn_toprank.csv',index=False)
    sir_pd.to_csv(root+'sir_toprank.csv',index=False)
    
def draw_3D(data1,title,save_path=None):
   plt.figure(dpi=200)
   ax1 = plt.axes(projection='3d')  # 设置三维轴
   ax1.scatter3D(data1['K-core'], data1['Number of community'],data1['ND'],c=data1['SIR'])
   plt.title(title,fontsize=16,fontweight='bold')
   plt.xlabel('K-core',fontsize=14,fontweight='bold')
   plt.ylabel('Number of community',fontsize=14,fontweight='bold')
   plt.yticks(np.arange(min(data1['Number of community']),max(data1['Number of community'])+5,dtype=int))
   ax1.set_zlabel('ND',fontsize=14,fontweight='bold')
   #plt.colorbar(f1,)
   #plt.tight_layout()
   
   if save_path:
       plt.savefig(save_path)
   plt.show()

def Vc(G,community):
    vc = {}
    for node in list(G.nodes()):
        com_set = set({community[node]})
        for nei in list(G.adj[node]):
            if community[nei] not in com_set:
                com_set.add(community[nei])
        vc[node] = len(com_set)
    return vc

def get_model_list(G,label,L_list,batch_size=32,num_epochs=500,lr=0.001):
    models=[]
    for L in L_list:
        data = Embeddings.main1(G,L)
        train_loader = Utils.Get_DataLoader1(data,label,batch_size,L)
        model = Models.CNN1(L)
        model,_ = Utils.train_model(train_loader,model,num_epochs,lr,L)
        models.append(model)
    return models

def SIR(G,infected,beta=0.1,miu=1):
    """SIR model
    Input:
        G:原始网络
        infected:被感染的节点
        miu:恢复的概率
    return:
        re:模拟N次之后，该节点的平均感染规模
    
    """
    N = 1000
    re = 0
    
    while N > 0:
        inf = set(infected) # 初始的被感染节点集合
        R = set() # 恢复的节点
        while len(inf) != 0:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    k = random.uniform(0,1)
                    if (k < beta) and (j not in inf) and (j not in R):
                        newInf.append(j)
                k2 = random.uniform(0, 1)
                if k2 >miu:
                    newInf.append(i)
                else:
                    R.add(i)
            inf = set(newInf)
        re += len(R)+len(inf)
        N -= 1
    return re/1000.0

def SIR_dict(G,beta=0.1,miu=1,real_beta=None,a=1.5):
    """获得整个网络的所有节点的SIR结果
    Input:
        G:目标网络
        beta:传播概率
        miu:恢复概率
        real_beta:按公式计算的传播概率
    return:
        SIR_dic:记录所有节点传播能力的字典
    """
    
    node_list = list(G.nodes())
    SIR_dic = {}
    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
    
        beta = a*(float(dc_list.mean())/(float((dc_list**2).mean())-float(dc_list.mean())))
    print('beta:',beta)
    for node in tqdm(node_list):
        sir = SIR(G,infected=[node],beta=beta,miu=miu)
        SIR_dic[node] = sir
    return SIR_dic

def save_sir_dict(dic,path):
    """存放SIR的结果
    Parameters:
        dic:sir结果(dict)
        path:目标存放路径
    """
    node = list(dic.keys())
    sir = list(dic.values())
    Sir = pd.DataFrame({'Node':node,'SIR':sir})
    Sir.to_csv(path,index=False)

def SIR_betas(G,a_list,root_path):
    """不同beta情况下的SIR
    Parameters:
        G:目标网络
        a_list:存放传播概率是传播阈值的多少倍的列表
    """
    sir_list = []
    for inx,a in enumerate(a_list):
        sir_dict = SIR_dict(G,real_beta=True,a=a)
        sir_list.append(sir_dict)
        path = root_path+str(inx)+'.csv'
        save_sir_dict(sir_dict,path)
    return sir_list