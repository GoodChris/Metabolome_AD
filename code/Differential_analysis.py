# -*- coding: utf-8 -*-
"""
@author: Xiaohui Luo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2,venn2_circles
import math
import PyComplexHeatmap
from PyComplexHeatmap import *
import warnings
warnings.filterwarnings("ignore")



def HV_plot(FCP,title):
    FC=FCP.loc[:,['FC']]
    Pvalue=FCP.loc[:,['FDR']]
    result=pd.DataFrame()
    result['x']=FC.apply(lambda x:math.log(x,2),axis=1)
    result['y']=Pvalue.apply(lambda x:-math.log(x,10),axis=1)
    x_threshold=math.log(1.2,2)
    y_threshold=-math.log(0.05,10)    
    result['group'] = 'dimgrey'
    result.loc[(result.x > x_threshold)&(result.y > y_threshold),'group'] = 'tab:red'
    result.loc[(result.x < -x_threshold)&(result.y > y_threshold),'group'] = 'tab:blue' 
    result.loc[result.y < y_threshold,'group'] = 'dimgrey'
    Sign={'tab:red':'Up','tab:blue':'Down','dimgrey':'Non-significant'}
    result['sign']=result['group'].apply(lambda x :Sign[x])
    xmin=-2
    xmax=2
    ymin=0
    ymax=6
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    dfg=result.groupby('sign')
    for k,v in dfg:
        ax.scatter(v['x'], v['y'], s=40, c=v['group'])
    ax.set_ylabel('-Log10 (FDR)',fontweight='bold')
    ax.set_xlabel('Log2 (Fold change)',fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.vlines(-x_threshold, ymin, ymax, color='dimgrey',linestyle='dashed', linewidth=1.5)
    ax.vlines(x_threshold, ymin, ymax, color='dimgrey',linestyle='dashed', linewidth=1.5)
    ax.hlines(y_threshold, xmin, xmax, color='dimgrey',linestyle='dashed', linewidth=1.5)
    legend_labels_colors = ['tab:red', 'tab:blue', 'dimgrey']
    legend_labels=[Sign[x] for x in legend_labels_colors]
    legend_handles_colors = [plt.Line2D([], [], marker='o', markersize=7, color=color, linestyle='None') for color in legend_labels_colors]  # 创建图例中的标记
    plt.legend(legend_handles_colors, legend_labels, bbox_to_anchor=(1.45,1))
    plt.title(title)
    plt.savefig('../PIC/VHs_%s.pdf'%title,dpi=300,format='pdf',bbox_inches='tight')
    plt.show()
    
def Ven(FCP_ad,FCP_mci):
    ad=FCP_ad[FCP_ad['FDR']<0.05]
    mci=FCP_mci[FCP_mci['FDR']<0.05]
    print("show counts")
    print("AD vs NC:",ad.shape[0])
    print("MCI vs NC",mci.shape[0])
    
    print("AD vs NC with FC>1.2:",ad[(ad['FC']>1.2)|(ad['FC']<1/1.2)].shape[0])
    print("MCI vs NC with FC>1.2:",mci[(mci['FC']>1.2)|(mci['FC']<1/1.2)].shape[0])
    
    adlist=ad[(ad['FC']>1.2)|(ad['FC']<1/1.2)].index.tolist()
    mcilist=mci[(mci['FC']>1.2)|(mci['FC']<1/1.2)].index.tolist()
    
    plt.figure()
    v=venn2(subsets=[set(adlist),set(mcilist)],set_labels=['AD vs NC','MCI vs NC'],
            set_colors=['r','orange'],alpha=0.6)
    c = venn2_circles(subsets=[set(adlist),set(mcilist)],alpha=0.8,linewidth=1.5, linestyle='dashed')
    v.get_patch_by_id('11').set_color('teal')
    plt.savefig('../PIC/Venn.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()
    


def Clustermap(data,info,lp=['AD','NC']):
    
    data=data[data['label'].isin(lp)]
    anno=info.loc[:,['Index','KEGG class','sign']]
    anno['UD']=anno['sign'].apply(lambda x : {'Lower in %s'%lp[0]:'Decreased','Higher in %s'%lp[0]:'Increased'}[x])
    anno=anno.set_index('Index')
    Ud=anno['UD']
    df_row=anno['KEGG class']
    row_ha = HeatmapAnnotation(Significant=anno_simple(Ud,cmap='Set2',height=5,legend=True,legend_kws=dict(fontsize=15,frameon=False,bbox_to_anchor=(1.3,1),loc=1,borderaxespad=0,title_fontsize=18)),
                               Class=anno_simple(df_row,cmap='tab20',legend=True,add_text=False,height=5,legend_kws=dict(fontsize=15,frameon=False,bbox_to_anchor=(1.3,1),loc=1,borderaxespad=0,title_fontsize=18)),
                               label_side='top',label_kws={'rotation':90,'rotation_mode':'anchor','color':'black'},axis=0)
    data=data.set_index('Subject')
    df_col=data.label
    if lp[0] == 'AD':
        color={"AD":"red","NC":"dodgerblue"}
    else:
        color={"NC":"dodgerblue","MCI":"orange"}
    col_ha = HeatmapAnnotation(Group=anno_simple(df_col,colors=color,legend=True,height=5,
                                                 legend_kws=dict(fontsize=15,frameon=False,bbox_to_anchor=(1.3,1),loc=1,borderaxespad=0,title_fontsize=18),add_text=False),
                               label_side='right',axis=1)
    flist=info['Index'].values.tolist()
    data=data.loc[:,flist]    
    data=data.T
    data=data.astype(float)
    plt.figure(figsize=(17, 9))
    cm = ClusterMapPlotter(data=data,z_score=None,standard_scale=None,
                           col_cluster=True,row_cluster=True,
                           col_cluster_method='centroid',col_cluster_metric='euclidean',
                           row_cluster_method='centroid',row_cluster_metric='euclidean',
                           top_annotation=col_ha,left_annotation=row_ha,
                           col_split=df_col, col_split_gap=2.5,
                           row_split=Ud,row_split_gap=1.5,
                           label='values',row_dendrogram=True,col_dendrogram=False,
                           show_rownames=False,show_colnames=False,
                           cmap='RdBu_r',legend=True)
    for cbar in cm.cbars:
        if isinstance(cbar,matplotlib.colorbar.Colorbar):
            cbar.outline.set_color('white')
            cbar.outline.set_linewidth(2)
            cbar.dividers.set_color('red')
            cbar.dividers.set_linewidth(2)
            cbar.set_ticks(np.arange(-4,5,1))
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.set_position([1.05,0.15,0.015,0.3])
        
    plt.savefig('../PIC/Heatmap_%s%s.pdf'%(lp[0],lp[1]),dpi=150,format='pdf',bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    
    FCP_ad=pd.read_csv('../data/FCP_ad.csv')
    FCP_mci=pd.read_csv('../data/FCP_mci.csv')
    
    HV_plot(FCP_ad,title='AD vs NC')
    HV_plot(FCP_mci,title='MCI vs NC')
    Ven(FCP_ad,FCP_mci)
    
    #adinfo=pd.read_csv('../data/FCP_ad_info.csv')
    #mciinfo=pd.read_csv('../data/FCP_mci_info.csv')
    #discoveryAD=pd.read_csv('../data/discovery_AD.csv')
    #discoveryMCI=pd.read_csv('../data/discovery_MCI.csv')
    
    #Clustermap(data=discoveryAD,info=adinfo,lp=['AD','NC'])
    #Clustermap(data=discoveryMCI,info=mciinfo,lp=['MCI','NC'])
