#!/usr/bin/env python 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import time
import tabulate


def preprocess_data(df):

    # Convert label to categorical
    df['label'] = df['label'].astype('category')

    # Replace NA for -1
    df = df.fillna(-1)

    # Convert object columns on categorical
    object_cols = df.columns.to_series().groupby(df.dtypes) \
                    .groups[np.dtype(np.object)]
    # Remove the 'id' col
    for col in object_cols[1:]:
        df[col] = df[col].astype('category')
 
    # Convert to dummies all 'object' except id
    df = pd.get_dummies(df, columns=object_cols[1:], prefix=object_cols[1:])

    # Set the 'label' column at the end of the Data Frame
    column_order = df.columns.tolist()
    column_order.pop( column_order.index('label') )
    column_order.append('label')
    df = df[column_order]
    
    return df



def pca_analysis(df):
    print("Running PCA analysis")
    start = time.time()
    pca = sklearn.decomposition.PCA()
    pca_cols = df.columns.tolist()[1:-1]
    pca.fit(df[pca_cols])
    end = time.time()
    
    print("PCA analysis finished in %.2f." % (end-start))
    variance_explained_pct = np.cumsum(pca.explained_variance_ratio_)
    variance_explained_pct_list = [ (x+1, variance_explained_pct[x]) 
                                     for x 
                                     in range(len(variance_explained_pct))
                                  ]
    variance_explained_table = tabulate.tabulate(variance_explained_pct_list)
    print("Accumulated variance explained by component:\n%s" % 
      variance_explained_table)

    transf_not_same = pca.transform(df[df.label==-1][pca_cols])
    transf_same = pca.transform(df[df.label==1][pca_cols])
  
    plt.figure()  
    plt.scatter( list(map((lambda x: x[0]), transf_not_same)),
                list(map((lambda x: x[1]), transf_not_same)),
                c='r',
                label='Different location'
        )
    plt.scatter( list(map((lambda x: x[0]), transf_same)),
                list(map((lambda x: x[1]), transf_same)),
                c='b',
                label='Same location'
        )
    plt.legend()
    plt.title("Scatterplot of PC1 vs PC2")

    plt.figure()
    plt.scatter( list(map((lambda x: x[0]), transf_not_same)),
                list(map((lambda x: x[2]), transf_not_same)),
                c='r',
                label='Different location'
        )
    plt.scatter( list(map((lambda x: x[0]), transf_same)),
                list(map((lambda x: x[2]), transf_same)),
                c='b',
                label='Same location'
        )
    plt.legend()
    plt.title("Scatterplot of PC1 vs PC3")

    plt.show()




if __name__ == '__main__':
    df = pd.read_csv('./data/header.csv', header=0, na_values='?')

    print("Running Preprocess...")
    start = time.time()
    df_dummy = preprocess_data(df)
    end = time.time()
    print("Preprocess finished in %.2f." % (end-start))

    pca_analysis(df_dummy)

    
# vim: set expandtab ts=4 sw=4: 
