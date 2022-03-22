# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:05:12 2021

@author: OuYang
"""

# 1 导入库
import numpy as np
import networkx as nx
import torch
import Utils

# RCNN输入构造主程序
def main(G,L):
    """节点输入构造主程序
    Parameters:
        G: 目标网络
        L: 嵌入矩阵的大小（包含目标节点的邻居网络节点总数）
    return:
        data_dict:存储每个节点嵌入矩阵的字典{v1:matrix_v1,...,v2:matrix_v2,...}
    """
    data_dict = {}
    node_list = list(G.nodes()) # 获得网络中的所有节点
    #对每个节点按照规则提取L-1个邻居节点
    for node in node_list:
        subset = [node] # 目标节点+固定数量的邻居节点
        one_order = list(G.adj[node]) #先看一阶邻居节点
        one_degree = dict(G.degree(one_order)) #获取一阶邻居节点的度值
        if len(one_order) >= L-1: #如果一阶邻居节点够了，那就不看二阶邻居了
            selected_degree = [len(one_order)] # 所选节点在原始网络的邻居数量
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)] # 按度值对一阶邻居排序
            for nei in selected_nei:
                if (nei not in subset) and (len(subset)<L):
                    subset.append(nei)
                    selected_degree.append(one_degree[nei])
            node_subgraph,node_A = Utils.generate_subgraph(G,subset) # 生成阶邻矩阵
            node_B = Utils.transform(node_A,selected_degree) # 转换
            data_dict[node] = node_B
        
        elif (len(one_order)< L-1) and (len(one_order)!=0): # 当一阶邻居节点不够并且一阶邻居数量不为0的时候，找更高阶的邻居
            selected_degree = [len(one_order)]
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)]
            gap = (L-1)-len(selected_nei) # 看看还差多少
            high_nei = set(selected_nei) # 高阶邻居节点
            neis = selected_nei
            count = 0 # 尝试50次，如果超过了50次就用padding
            while True:
                if count==50:
                    break
                new_order = set([])
                for nei in neis: # 遍历每个邻居节点的邻居
                    nei_nei = list(G.adj[nei])
                    for each in nei_nei:
                        if (each != node) and (each not in high_nei):
                            new_order.add(each)
                new_order_list = list(new_order)
                degree_new = dict(G.degree(new_order_list))
                new_selected_nei = [i for i,j in sorted(degree_new.items(),key=lambda x:x[1],reverse=True)]
                if len(new_selected_nei) >=gap: # 满足了数量
                    for i in range(gap):
                        selected_nei.append(new_selected_nei[i])
            
                    break
                
                elif len(new_selected_nei)<gap: # 没满足
                    for new in new_selected_nei:
                        selected_nei.append(new)
                        
                        gap-=1
                    
                    neis = new_order_list
                    for each in neis:
                        high_nei.add(each)
                count+=1

            for neii in selected_nei:
                if neii not in subset:
                    subset.append(neii)
                    selected_degree.append(len(G.adj[neii]))
            padding = L-len(subset)
            node_subgraph,node_A = Utils.generate_subgraph(G,subset)
            node_B = Utils.transform(node_A,selected_degree)
            if padding == 0:
                data_dict[node] = node_B
            else:
                node_B_padding = np.zeros([L,L])
                for row in range(node_B.shape[0]):
                    node_B_padding[row,:node_B.shape[0]] = node_B[row,:]
                data_dict[node] = node_B_padding
        else: #当节点为孤立节点时，直接用一个L*L的零矩阵来表示
            data_dict[node] = np.zeros([L,L])
    return data_dict

# M-RCNN输入构造主程序
def main1(G,L,community):
    """节点嵌入生成主程序
    Parameters:
        G: 目标网络
        L: 嵌入矩阵的大小（包含目标节点的邻居网络节点总数）
    return:
        data_dict:存储每个节点嵌入矩阵的字典{v1:matrix_v1,...,v2:matrix_v2,...}
    """
    data_dict = {}
    node_list = list(G.nodes()) # 获得网络中的所有节点
    #对每个节点按照规则提取L-1个邻居节点
    k_shell = dict(nx.core_number(G))
    nd = Utils.neighbor_degree(G)
    for node in node_list:
        subset = [node] # 目标节点+固定数量的邻居节点
        one_order = list(G.adj[node]) #先看一阶邻居节点
        one_degree = dict(G.degree(one_order)) #获取一阶邻居节点的度值
        if len(one_order) >= L-1: #如果一阶邻居节点够了，那就不看二阶邻居了
            selected_com = [community[node]] # 所选节点在原始网络的邻居数量
            selected_shell = [k_shell[node]]
            selected_nd = [nd[node]]
            
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)] # 按度值对一阶邻居排序
            for nei in selected_nei:
                if (nei not in subset) and (len(subset)<L):
                    subset.append(nei)
                    selected_com.append(community[nei])
                    selected_shell.append(k_shell[nei])
                    selected_nd.append(nd[nei])
            
            node_subgraph,node_A = Utils.generate_subgraph(G,subset) # 生成阶邻矩阵
            node_B = Utils.transform1(node_A,selected_nd,selected_com,selected_shell) # 转换
            data_dict[node] = node_B
        
        elif (len(one_order)< L-1) and (len(one_order)!=0): # 当一阶邻居节点不够并且一阶邻居数量不为0的时候，找更高阶的邻居
            selected_com = [community[node]]
            selected_shell = [k_shell[node]]
            selected_nd = [nd[node]]
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)]
            gap = (L-1)-len(selected_nei) # 看看还差多少
            high_nei = set(selected_nei) # 高阶邻居节点
            neis = selected_nei
            count = 0 # 尝试50次，如果超过了50次就用padding
            while True:
                if count==50:
                    break
                new_order = set([])
                for nei in neis: # 遍历每个邻居节点的邻居
                    nei_nei = list(G.adj[nei])
                    for each in nei_nei:
                        if (each != node) and (each not in high_nei):
                            new_order.add(each)
                new_order_list = list(new_order)
                degree_new = dict(G.degree(new_order_list))
                new_selected_nei = [i for i,j in sorted(degree_new.items(),key=lambda x:x[1],reverse=True)]
                if len(new_selected_nei) >=gap: # 满足了数量
                    for i in range(gap):
                        selected_nei.append(new_selected_nei[i])
                    break
                
                elif len(new_selected_nei)<gap: # 没满足
                    for new in new_selected_nei:
                        selected_nei.append(new)
                        gap-=1
                    neis = new_order_list
                    for each in neis:
                        high_nei.add(each)
                count+=1

            for neii in selected_nei:
                if neii not in subset:
                    subset.append(neii)
                    selected_com.append(community[neii])
                    selected_shell.append(k_shell[neii])
                    selected_nd.append(nd[neii])
            padding = L-len(subset)
            node_subgraph,node_A = Utils.generate_subgraph(G,subset)
            node_B = Utils.transform1(node_A,selected_nd,selected_com,selected_shell)
            if padding == 0:
                data_dict[node] = node_B
            else:
                node_B_padding = torch.zeros([3,L,L])
                for row in range(node_B.shape[0]):
                    node_B_padding[:,:node_B.shape[1],:node_B.shape[1]] = node_B
                data_dict[node] = node_B_padding
        else: #当节点为孤立节点时，直接用一个L*L的零矩阵来表示
            data_dict[node] = torch.zeros([3,L,L])
    return data_dict

def main2(G,L,community,method):
    """节点嵌入生成主程序
    Parameters:
        G: 目标网络
        L: 嵌入矩阵的大小（包含目标节点的邻居网络节点总数）
    return:
        data_dict:存储每个节点嵌入矩阵的字典{v1:matrix_v1,...,v2:matrix_v2,...}
    """
    data_dict = {}
    node_list = list(G.nodes()) # 获得网络中的所有节点
    #对每个节点按照规则提取L-1个邻居节点
    k_shell = dict(nx.core_number(G))
    nd = neighbor_degree(G)
    for node in node_list:
        subset = [node] # 目标节点+固定数量的邻居节点
        one_order = list(G.adj[node]) #先看一阶邻居节点
        one_degree = dict(G.degree(one_order)) #获取一阶邻居节点的度值
        if len(one_order) >= L-1: #如果一阶邻居节点够了，那就不看二阶邻居了
            selected_com = [community[node]] # 所选节点在原始网络的邻居数量
            selected_shell = [k_shell[node]]
            selected_nd = [nd[node]]
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)] # 按度值对一阶邻居排序
            for nei in selected_nei:
                if (nei not in subset) and (len(subset)<L):
                    subset.append(nei)
                    selected_com.append(community[nei])
                    selected_shell.append(k_shell[nei])
                    selected_nd.append(nd[nei])
            
            node_subgraph,node_A = Utils.generate_subgraph(G,subset) # 生成阶邻矩阵
            if method == 'community':
                node_B = Utils.transform2(node_A,selected_nd,selected_com) # 转换
            else:
                node_B = Utils.transform2(node_A,selected_nd,selected_shell) 
            data_dict[node] = node_B
        
        elif (len(one_order)< L-1) and (len(one_order)!=0): # 当一阶邻居节点不够并且一阶邻居数量不为0的时候，找更高阶的邻居
            selected_com = [community[node]]
            selected_shell = [k_shell[node]]
            selected_nd = [nd[node]]
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)]
            gap = (L-1)-len(selected_nei) # 看看还差多少
            high_nei = set(selected_nei) # 高阶邻居节点
            neis = selected_nei
            count = 0 # 尝试50次，如果超过了50次就用padding
            while True:
                if count==50:
                    break
                new_order = set([])
                for nei in neis: # 遍历每个邻居节点的邻居
                    nei_nei = list(G.adj[nei])
                    for each in nei_nei:
                        if (each != node) and (each not in high_nei):
                            new_order.add(each)
                new_order_list = list(new_order)
                degree_new = dict(G.degree(new_order_list))
                new_selected_nei = [i for i,j in sorted(degree_new.items(),key=lambda x:x[1],reverse=True)]
                if len(new_selected_nei) >=gap: # 满足了数量
                    for i in range(gap):
                        selected_nei.append(new_selected_nei[i])
                    break
                
                elif len(new_selected_nei)<gap: # 没满足
                    for new in new_selected_nei:
                        selected_nei.append(new)
                        gap-=1
                    neis = new_order_list
                    for each in neis:
                        high_nei.add(each)
                count+=1

            for neii in selected_nei:
                if neii not in subset:
                    subset.append(neii)
                    selected_com.append(community[neii])
                    selected_shell.append(k_shell[neii])
                    selected_nd.append(nd[neii])
            padding = L-len(subset)
            node_subgraph,node_A = Utils.generate_subgraph(G,subset)
            if method == 'community':
                node_B = Utils.transform2(node_A,selected_nd,selected_com)
            else:
                node_B = Utils.transform2(node_A,selected_nd,selected_shell)
            if padding == 0:
                data_dict[node] = node_B
            else:
                node_B_padding = torch.zeros([2,L,L])
                for row in range(node_B.shape[0]):
                    node_B_padding[:,:node_B.shape[1],:node_B.shape[1]] = node_B
                data_dict[node] = node_B_padding
        else: #当节点为孤立节点时，直接用一个L*L的零矩阵来表示
            data_dict[node] = torch.zeros([2,L,L])
    return data_dict