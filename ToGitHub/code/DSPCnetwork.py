# -*- coding: utf-8 -*-
"""
@author: Xiaohui Luo
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


def creat_net(X,Ninfo):
    node1=X['node1'].values.tolist()
    node2=X['node2'].values.tolist()
    weight=X['coef'].values.tolist()
    nodes=Ninfo['Compounds'].values.tolist()
    node_colors=Ninfo['KEGG class'].apply(lambda x : {'Lipids':'steelblue','Others':'lightskyblue',
                                                      'Peptide':'darkorange','Nucleotide':'goldenrod',
                                                      'Cofactor/vitamin':'pink',
                                                      'Amino acid':'crimson','Xenobiotics':'green'}[x]).values.tolist()
    g=nx.Graph()
    g.add_nodes_from(nodes)
    
    for i in range(len(node1)):
        g.add_edge(node1[i],node2[i],weight=weight[i])
    plt.figure(figsize=(12,10))
    pos=nx.random_layout(g)
    edge_weight=[g.edges[i,j]['weight'] for i,j in g.edges]
    nx.draw_networkx_nodes(g, pos=pos, nodelist=nodes, node_color=node_colors,node_size = 650,alpha=0.9)
    nx.draw_networkx_edges(g, pos=pos, edge_color=edge_weight,width=[10*abs(x) for x in edge_weight], edge_cmap=plt.cm.seismic,alpha=0.9)     
    nx.draw_networkx_labels(g,pos,font_size=12)
    pc = matplotlib.collections.PatchCollection([], cmap=plt.cm.seismic)
    pc.set_array(edge_weight)
    cbar=plt.colorbar(pc,fraction=0.02,pad=0.08)
    cbar.set_label('Coefficient')
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig('../PIC/DSPCnetwork.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()
    
    
    
if __name__=='__main__':
    
    X=pd.read_csv('../data/networkadjp005.csv')
    Ninfo=pd.read_csv('../data/nodeinfo.csv')
    creat_net(X,Ninfo)
