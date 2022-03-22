# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:43:57 2021

@author: OuYang
"""
# In[1]
# 1 导入库
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Models
import Utils
import Test
import Embeddings
sns.set_style('ticks')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# In[2]
if __name__ == '__main__':
# In[3]
    # 固定随机种子
    Utils.setup_seed(5)
    # 导入网络
    # 生成人工网络的标签
    BA_1000_4 = Utils.load_graph('./训练人工网络/BA_1000_4.txt')
    BA_1000_10 = Utils.load_graph('./训练人工网络/BA_1000_10.txt')
    BA_1000_20 = Utils.load_graph('./训练人工网络/BA_1000_20.txt')
    
    BA_2000_4 = Utils.load_graph('./测试人工网络/BA_2000_4.txt')
    BA_2000_10 = Utils.load_graph('./测试人工网络/BA_2000_10.txt')
    BA_2000_20 = Utils.load_graph('./测试人工网络/BA_2000_20.txt')

    BA_3000_4 = Utils.load_graph('./训练人工网络/BA_3000_4.txt')
    BA_3000_10 = Utils.load_graph('./训练人工网络/BA_3000_10.txt')
    BA_3000_20 = Utils.load_graph('./训练人工网络/BA_3000_20.txt')
    
    # 网络中的节点标签
    BA_1000_4_label = Utils.SIR_dict(BA_1000_4,real_beta=True)
    BA_1000_10_label = Utils.SIR_dict(BA_1000_10,real_beta=True)
    BA_1000_20_label = Utils.SIR_dict(BA_1000_20,real_beta=True)
    
    BA_2000_4_label = Utils.SIR_dict(BA_2000_4,real_beta=True)
    BA_2000_10_label = Utils.SIR_dict(BA_2000_10,real_beta=True)
    BA_2000_20_label = Utils.SIR_dict(BA_2000_20,real_beta=True)

    BA_3000_4_label = Utils.SIR_dict(BA_3000_4,real_beta=True)
    BA_3000_10_label = Utils.SIR_dict(BA_3000_10,real_beta=True)
    BA_3000_20_label = Utils.SIR_dict(BA_3000_20,real_beta=True)
    
    BA_1000_4_pd=pd.DataFrame({'Nodes':list(BA_1000_4_label.keys()),'SIR':list(BA_1000_4_label.values())})
    BA_1000_10_pd=pd.DataFrame({'Nodes':list(BA_1000_10_label.keys()),'SIR':list(BA_1000_10_label.values())})
    BA_1000_20_pd=pd.DataFrame({'Nodes':list(BA_1000_20_label.keys()),'SIR':list(BA_1000_20_label.values())})
    BA_2000_4_pd=pd.DataFrame({'Nodes':list(BA_2000_4_label.keys()),'SIR':list(BA_2000_4_label.values())})
    BA_2000_10_pd=pd.DataFrame({'Nodes':list(BA_2000_10_label.keys()),'SIR':list(BA_2000_10_label.values())})
    BA_2000_20_pd=pd.DataFrame({'Nodes':list(BA_2000_20_label.keys()),'SIR':list(BA_2000_20_label.values())})
    BA_3000_4_pd=pd.DataFrame({'Nodes':list(BA_3000_4_label.keys()),'SIR':list(BA_3000_4_label.values())})
    BA_3000_10_pd=pd.DataFrame({'Nodes':list(BA_3000_10_label.keys()),'SIR':list(BA_3000_10_label.values())})
    BA_3000_20_pd=pd.DataFrame({'Nodes':list(BA_3000_20_label.keys()),'SIR':list(BA_3000_20_label.values())})    
        
    # In[2] 生成真实网络的标签
    PowerGrid = Utils.load_graph('./实际网络/powergrid.txt')
    GrQ = Utils.load_graph('./实际网络/CA-GrQc.txt')
    Facebook = Utils.load_graph('./实际网络/facebook_combined.txt')
    Ham = Utils.load_graph('./实际网络/Peh_edge.txt')
    Hep = Utils.load_graph('./实际网络/CA-HepTh.txt')
    LastFM = Utils.load_graph('./实际网络/LastFM.txt')
    Figeys = Utils.load_graph('./实际网络/figeys.txt')
    Vidal = Utils.load_graph('./实际网络/vidal.txt')
    Sex = Utils.load_graph('./实际网络/Sex.txt')
    
    a_list = np.arange(1.0,2.0,0.1)
    Facebook_SIR = Utils.SIR_betas(Facebook,a_list,root_path4)
    Sex_SIR = Utils.SIR_betas(Sex,a_list,root_path5)
    Vidal_SIR = Utils.SIR_betas(Vidal,a_list,root_path6)
    Figeys_SIR = Utils.SIR_betas(Figeys,a_list,root_path10)
    Ham_SIR = Utils.SIR_betas(Ham,a_list,root_path11)
    Hep_SIR = Utils.SIR_betas(Hep,a_list,root_path12)
    LastFM_SIR = Utils.SIR_betas(LastFM,a_list,root_path13)
    PowerGrid_SIR = Utils.SIR_betas(PowerGrid,a_list,root_path1)
    GrQ_SIR = Utils.SIR_betas(GrQ,a_list,root_path3)