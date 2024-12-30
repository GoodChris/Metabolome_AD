# -*- coding: utf-8 -*-
"""
@author: Xiaohui Luo
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation



def gbox_plotP(data,tf):
    df=data.loc[:,['label','Dataset',tf]]
    order=['NC','MCI','AD']
    color=['dodgerblue','orange','red']
    #plt.figure(figsize=(4,5))
    ax=sns.boxplot(data=df,x='Dataset',y=tf,hue='label',hue_order=order,palette=color,width=0.75,saturation=1)
    add_stat_annotation(ax,data=df,x='Dataset',y=tf,hue='label',hue_order=order,
                        box_pairs=[(('Discovery','NC'),('Discovery','MCI')),(('Discovery','MCI'),('Discovery','AD')),
                                   (('Discovery','NC'),('Discovery','AD')),(('Replication','NC'),('Replication','MCI')),
                                   (('Replication','MCI'),('Replication','AD')),(('Replication','NC'),('Replication','AD'))],
                        test='t-test_ind',text_format='star',comparisons_correction=None)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Discovery','Replication'],fontsize=15)
    #ax.set_yticks(np.arange(-4,8),fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_ylabel('Z-score',fontsize=20)
    ax.set_title(tf,fontsize=20)
    ax.set_xlabel("")
    plt.legend([],[], fontsize=10,frameon=False)
    plt.legend(fontsize=10,bbox_to_anchor=(1.25,1),loc=1,borderaxespad=0,title='label',title_fontsize=15)
    
    plt.show()
        
        
        
        
if __name__=='__main__':
    
    
    Pdata=pd.read_csv('../data/Pathway_data.csv')
    for tf in Pdata.columns[2:-1]:
         gbox_plotP(Pdata,tf)