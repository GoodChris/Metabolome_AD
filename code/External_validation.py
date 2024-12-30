# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:20:49 2024

@author: Shawhii
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from sklearn.decomposition import PCA
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def Jplot_PCA(D):
    data=D.drop(['Subject','label','Dataset'],axis=1)
    pca = PCA(n_components=2) 
    X=np.array(data)
    X=pca.fit_transform(X)
    
    Dg=D.loc[:,['Subject','label']]
    Dg['PC1'],Dg['PC2']=X[:,0],X[:,1]
    dfg = Dg.groupby('label')
    plt.figure(figsize=(4,7))
    fig,ax=plt.subplots()
    for k,v in dfg:
        x,y=v['PC1'].values.tolist(),v['PC2'].values.tolist()
        plt.scatter(x,y,marker='.',c=[{"AD":"red","NC":"dodgerblue","MCI":"orange"}[t] for t in v['label'].values.tolist()],
                    label=k,s=50)
        cov_m=np.cov(np.array(v.loc[:,['PC1','PC2']]).T)
        center=np.mean(np.array(v.loc[:,['PC1','PC2']]),axis=0)
        e1,e2=np.linalg.eig(cov_m)
        angle=np.degrees(np.arctan2(*e2[:,0][::-1]))
        width=2*np.sqrt(e1[0]*np.sqrt(chi2.ppf(0.95,df=2)))
        height=2*np.sqrt(e1[1]*np.sqrt(chi2.ppf(0.95,df=2)))
        ellipse=Ellipse(xy=center, width=width, height=height, angle=angle, 
                        edgecolor={"AD":"red","NC":"dodgerblue","MCI":"orange"}[k],facecolor='none')
        ax.add_patch(ellipse)
    ax.set_xlabel("PC1(%.2f%%)"%(100*pca.explained_variance_ratio_[0]),fontsize=20)
    ax.set_ylabel("PC2(%.2f%%)"%(100*pca.explained_variance_ratio_[1]),fontsize=20)
    ax.set_title("External validation")
    ax.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0)
    plt.savefig('../PIC/PCA_extenalvalidation1.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()
    
def Pplot_PCA(D):
    D=D[D['label'].isin(['Dementia','HE'])]
    data=D.drop(['Subject','label','Dataset'],axis=1)
    pca = PCA(n_components=2) 
    X=np.array(data)
    X=pca.fit_transform(X)
    
    Dg=D.loc[:,['Subject','label']]
    Dg['PC1'],Dg['PC2']=X[:,0],X[:,1]
    dfg = Dg.groupby('label')
    plt.figure(figsize=(4,7))
    fig,ax=plt.subplots()
    for k,v in dfg:
        x,y=v['PC1'].values.tolist(),v['PC2'].values.tolist()
        plt.scatter(x,y,marker='.',c=[{"Dementia":"deeppink","HE":"deepskyblue","HY":"orange"}[t] for t in v['label'].values.tolist()],
                    label=k,s=50)
        cov_m=np.cov(np.array(v.loc[:,['PC1','PC2']]).T)
        center=np.mean(np.array(v.loc[:,['PC1','PC2']]),axis=0)
        e1,e2=np.linalg.eig(cov_m)
        angle=np.degrees(np.arctan2(*e2[:,0][::-1]))
        width=2*np.sqrt(e1[0]*np.sqrt(chi2.ppf(0.95,df=2)))
        height=2*np.sqrt(e1[1]*np.sqrt(chi2.ppf(0.95,df=2)))
        ellipse=Ellipse(xy=center, width=width, height=height, angle=angle, 
                        edgecolor={"Dementia":"deeppink","HE":"deepskyblue","HY":"orange"}[k],facecolor='none')
        ax.add_patch(ellipse)
    ax.set_xlabel("PC1(%.2f%%)"%(100*pca.explained_variance_ratio_[0]),fontsize=20)
    ax.set_ylabel("PC2(%.2f%%)"%(100*pca.explained_variance_ratio_[1]),fontsize=20)
    ax.set_title("External validation")
    ax.legend(bbox_to_anchor=(0.98,0.98),loc=1,borderaxespad=0)
    plt.savefig('../PIC/PCA_extenalvalidation2.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()

def data_trans(data,tlist):
    D=data.loc[:,['Subject','label',tlist[0]]]
    D['pathway']=tlist[0]
    D=D.rename(columns={tlist[0]:'Values'})
    for ts in tlist[1:]:
        d=data.loc[:,['Subject','label',ts]]
        d['pathway']=ts
        d=d.rename(columns={ts:'Values'})
        D=D.append(d)
    return D 

def GV_plotJ(data):
    tlist=data.columns[2:-1]
    #tlist=['Linoleic acid metabolism', 'Arachidonic acid metabolism', 'alpha-Linolenic acid metabolism']
    data=data_trans(data,tlist)
    order=['NC','MCI','AD']
    color=['dodgerblue','orange','red']
    plt.figure(figsize=(12,6))
    ax=sns.violinplot(data=data,x='pathway',y='Values',hue='label',hue_order=order,palette=color,width=0.75,saturation=1)
    add_stat_annotation(ax,data=data,x='pathway',y='Values',hue='label',hue_order=order,
                        box_pairs=[((x,'NC'),(x,'MCI')) for x in list(set(data['pathway'].values.tolist()))]+[((x,'MCI'),(x,'AD')) for x in list(set(data['pathway'].values.tolist()))]+[((x,'NC'),(x,'AD')) for x in list(set(data['pathway'].values.tolist()))],
                        test='t-test_ind',text_format='star',comparisons_correction=None)
    ax.set_xticklabels(tlist,fontsize=13,rotation=45,ha='right')
    ax.set_ylabel('Z-scored abundance',fontsize=20)
    ax.set_xlabel('pathway',fontsize=20)
    plt.legend(loc=3,title='label')
    plt.savefig('../PIC/Pathway_extenalvalidation1.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()
    
def GV_plotP(data):
    tlist=data.columns[2:-1]
    #tlist=['Caffeine metabolism', 'Ether lipid metabolism']
    data=data_trans(data,tlist)
    data=data[data['label'].isin(['Dementia','HE'])]
    order=['HE','Dementia']
    color=['deepskyblue','deeppink']
    plt.figure(figsize=(7,6))
    ax=sns.violinplot(data=data,x='pathway',y='Values',hue='label',hue_order=order,palette=color,width=0.75,saturation=1)
    add_stat_annotation(ax,data=data,x='pathway',y='Values',hue='label',hue_order=order,
                        box_pairs=[((x,'HE'),(x,'Dementia')) for x in list(set(data['pathway'].values.tolist()))],
                        test='t-test_ind',text_format='star',comparisons_correction=None)
    ax.set_xticklabels(tlist,fontsize=13,rotation=45,ha='right')
    ax.set_ylabel('Z-scored abundance',fontsize=20)
    ax.set_xlabel('pathway',fontsize=20)
    plt.legend(loc=3,title='label')
    plt.savefig('../PIC/Pathway_extenalvalidation2.pdf',dpi=150,format='pdf',bbox_inches='tight')
    plt.show()



if __name__=='__main__':
    
    
    Pdata1=pd.read_csv('../data/Pathway_externaldata1.csv')
    Jplot_PCA(D=Pdata1)
    GV_plotJ(data=Pdata1)

     
    Pdata2=pd.read_csv('../data/Pathway_externaldata2.csv')
    
    Pplot_PCA(D=Pdata2)
    GV_plotP(data=Pdata2)