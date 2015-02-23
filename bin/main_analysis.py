#!/usr/bin/env python 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tabulate
import pickle;

import sklearn.decomposition
import sklearn.cross_validation
import sklearn.lda
import sklearn.qda
import sklearn.svm
import sklearn.tree
import sklearn.linear_model
import sklearn.neighbors




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



def get_train_test_df(df, train_rate):
    '''Split a data frame in train and test'''

    rows = df_dummy.index.values
    np.random.shuffle(rows)

    train_rows = rows[:round(len(rows)*train_rate)]
    test_rows = rows[round(len(rows)*train_rate):]

    train_rows = df.loc[train_rows]
    test_rows = df.loc[test_rows]

    return (train_rows, test_rows)

 

def get_best_models(model_classes, model_params, 
                    x_train, x_test, y_train, y_test, 
                    cv=5, n_jobs=3):
    '''Get the best Lineal Discriminant Moldel'''

    ret = {}

    train_rate = len(df_train)/(len(df_train)+len(df_test))
    x_df = pd.concat((x_train, x_test))
    y = pd.concat((y_train, y_test))

    start_time = time.time()
    # iterate over all model classes
    for model_name in model_classes.keys():
        print("Starting to evaluate model %s." % model_name)
        class_name = model_classes[model_name]
        params_alternatives = model_params.get(model_name, [{}])
    
        # Train the model with all set of parameters to get the best 
        # cross-validated model
        best_acc_std = (0, 1000000)
        best_params = None
        best_scores = None
        for params in params_alternatives:
            start_train = time.time()
            print("Training model %s with params %s..." % (model_name, str(params)))
            
            model = class_name(**params)    # Instanciate the model
            # train it
            current_scores = sklearn.cross_validation.cross_val_score(
                model, x_df, y, cv=cv, n_jobs=n_jobs)
            end_train = time.time()
            
            scores_mean = current_scores.mean()
            scores_std = current_scores.std()
            print(('Model %s with params %s finished in %.2f secs. ' + 
                  'Accuracy: %0.2f (+/- %0.2f)') % 
                    (model_name, str(params), end_train-start_train, 
                     scores_mean, scores_std*2)
                 )

            # Check if it is the best model
            if (scores_mean > best_acc_std[0]) or \
               (scores_mean == best_acc_std[0] and 
                  scores_std < best_acc_std[1]):
                    best_acc_std = (scores_mean, scores_std)
                    best_params = params
                    best_scores = current_scores

        # Stores the info of the best param of the model
        ret[model_name] = {
                            'params': best_params,
                            'cv_scores': best_scores,
                          }


    # All models and params were modeled and cross-validated.
    # Now train the best model of each classifier.
    for model_name in ret.keys():
        start_train = time.time()
        print("Training final %s model with params %s." %
            (model_name, str(ret[model_name]['params'])))

        final_model = class_name(**best_params)
        final_model.fit(x_train, y_train)
        ret[model_name]['model'] = final_model
        
        end_train = time.time()
        print("Final %s model with params %s was done in %.2f secs" %
              (model_name, str(ret[model_name]['params']), 
               end_train - start_train))

    print("Training process finished in %.2f." % (time.time()-start_time))
    return ret
     

def  save_object(filename, obj):
    '''Save an object to a file'''

    fh = open(filename, 'wb')
    pickle.dump(obj, fh)
    fh.close()



if __name__ == '__main__':
    df = pd.read_csv('./data/header.csv', header=0, na_values='?')

    print("Running Preprocess...")
    start = time.time()
    df_dummy = preprocess_data(df)
    end = time.time()
    print("Preprocess finished in %.2f." % (end-start))

    #pca_analysis(df_dummy)

    # Get training and testing data structures
    df_train, df_test = get_train_test_df(df_dummy, 0.7)
    x_cols = df_dummy.columns.tolist()[1:-1]
    x_train = df_train[x_cols]
    x_test = df_test[x_cols]
    y_train = df_train.label
    y_test = df_test.label
    
    model_classes = {
#        'lda':  sklearn.lda.LDA,
#        'svm': sklearn.svm.SVC,
        'cart': sklearn.tree.DecisionTreeClassifier,
        'logistic regression': sklearn.linear_model.LogisticRegression,
#        'k-nn': sklearn.neighbors.KNeighborsClassifier,
        }

    model_params = {
        #'cart': [{'max_depth': x} for x in range(2,21)],
        'cart': [{'max_depth': x} for x in range(2,4)],
        'svm': [{'gamma': x} for x in (0, .1, .01, .001, .0001, .00001)],
        'k-nn': [{'n_neighbors': x} for x in range(3,21)],
        }
                
    models = get_best_models(model_classes, model_params, 
                             x_train, x_test, y_train, y_test)
    

    save_object('./data/models.pickle', models)
# vim: set expandtab ts=4 sw=4: 
