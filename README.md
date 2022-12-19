# Multi-channel-RCNN
This is a Pytorch implementation of Multi-channel RCNN for the task of (supervised) identification of spreading influence nodes in a graph, as described in our paper:

Ou Y, Guo Q, Xing J L, Liu J G, Identification of spreading influence nodes via multi-level structural attributes based on the graph convolutional network. ESWA, 2022, 117515 https://doi.org/10.1016/j.eswa.2022.117515 

The algorithm flowchart of M-RCNN is as follow:

![image](https://user-images.githubusercontent.com/67104283/168262812-e17e9a9b-d097-42d8-b590-85c5215f83e3.png)

**Notice:**
  The package `community` is required to run our code. If you have this problem, you can run:
  
  `pip uninstall community`
  
  `pip install python-louvain `
  
  `import community.community_louvain as community`

# Cite our paper if you use this code
```
@article{OU2022117515,
title = {Identification of spreading influence nodes via multi-level structural attributes based on the graph convolutional network},
journal = {Expert Systems with Applications},
volume = {203},
pages = {117515},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.117515},
url = {https://www.sciencedirect.com/science/article/pii/S0957417422008405},
author = {Yang Ou and Qiang Guo and Jia-Liang Xing and Jian-Guo Liu},
keywords = {Complex networks, Spreading influence, Graph convolutional network, Community structure, -core value},
abstract = {The network structural properties at the micro-level, community-level and macro-level have different contributions to the spreading influence of nodes. The challenge is how to make better use of different structural information while keeping the efficiency of the spreading influence identification algorithm. By taking the micro-level, community-level and macro-level structural information into account, an improved graph convolutional network based algorithm, namely the multi-channel RCNN (M-RCNN) is proposed to identify spreading influence nodes. As we focus on both the efficiency and accuracy of the algorithm, three centralities with low computational complexity are introduced: the sum of neighbors’ degree, the number of communities a node is connected with, and the k-core value. To construct the input of the M-RCNN, we first use the Breadth-first algorithm to extract a fixed-size neighborhood network for each node. Then exploit three matrices to encode the input of nodes rather than simply embedding different levels of structural information into the same matrix, which allows the weights that couple the three structural properties to be learned automatically during the training process. The experiments conducted on nine real-world networks show that, on average, compared with the RCNN algorithm, the accuracy obtained by the M-RCNN outperforms by 9.25%. By conducting efficiency test on nine Barabasi–Albert networks, the results show that the computational complexity of the M-RCNN is close to the RCNN. This work is helpful for deeply understanding the effects of network structure on the graph convolutional network performance.}
}
```
