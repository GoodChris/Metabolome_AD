# -*- coding: utf-8 -*-
"""
@author: Xiaohui Luo
"""

import pandas as pd


def pathway_map(PC_link,data):
    '''
    1.Metabolic pathways are quantified by summarizing the abundance of the measured metabolites 
    within the pathway to represent metabolic changes at the pathway level.
    2.The data file is a metabolomics data matrix with metabolic features as columns and samples as rows. 
    Metabolic features are encoded with KEGG ID, i.e. C number.
    3.The PC_link file contains the correspondence between pathways and metabolites, as provided below.
    '''
    Clist=data.drop(['Subject','label'],axis=1).columns.tolist()
    Pg=PC_link.groupby('pathway')
    Pathway_data=data.loc[:,['Subject','label']].copy()
    for k,v in Pg:
        clist=v['cpd'].values.tolist()
        clist=[x.split(':')[1] for x in clist]
        ovlist=list(set(clist).intersection(set(Clist)))
        if len(ovlist)==0:
            pass
        else:
            Pathway_data[k]=data.loc[:,ovlist].sum(axis=1).tolist()
    return Pathway_data



if __name__=='__main__':
    
    cpd_info=pd.read_csv('../data/cpd_info.txt',sep='\t')
    pathway_info=pd.read_csv('../data/pathway_info.txt',sep='\t')
    PC_link=pd.read_csv('../data/pathway_cpd.txt',sep='\t')