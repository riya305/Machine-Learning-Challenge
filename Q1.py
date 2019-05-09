import csv
import datetime
import math
import numpy as np
from sklearn.feature_selection import RFE
from sklearn import datasets, linear_model
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

def shifter(row):
    return np.hstack((np.delete(np.array(row), [3]), [np.nan]))

def main():

    #Load csv file into training list with CSV module
    train_dataset=list()
    dict_train={}
    with open('first-question/dataset/train_file.csv') as f:
        reader = csv.reader(f, delimiter = ',')
        cols=next(reader)
        for line in reader:
            if line != []:
                train_dataset.append(line)
                # dict_train[line[0]]=line[1:]

    #Load csv file into testing list with CSV module
    test_dataset=list()
    with open('first-question/dataset/test_file.csv') as f:
        reader = csv.reader(f, delimiter = ',')
        cols1=next(reader)
        for line in reader:
            if line != []:
                test_dataset.append(line)
                # dict_train[line[0]]=line[1:]
   
    df_train = pd.DataFrame(train_dataset,columns=cols)
    df_test = pd.DataFrame(test_dataset,columns=cols1)
    
    #Some rows in train_file.csv has misplaced values for particular columns which is 
    #corrected using the below piece of code
    mask = df_train['Contractor'] == 'Initial Information Collected'
    df_train.loc[mask, :] = df_train.loc[mask, :].apply(shifter, axis=1)

    # df_train['Issue Date'] = df_train['Issue Date'].astype('datetime64[ns]')
    # df_train['Application Date'] = df_train['Application Date'].astype('datetime64[ns]')

    # df_train['diff'] = (df_train['Issue Date'] - df_train['Application Date']).dt.days

    # df_test['Issue Date'] = df_test['Issue Date'].astype('datetime64[ns]')
    # df_test['Application Date'] = df_test['Application Date'].astype('datetime64[ns]')

    # df_test['diff'] = (df_test['Issue Date'] - df_test['Application Date']).dt.days

    # df_train['diff'].fillna(0, inplace=True)
    # df_test['diff'].fillna(0, inplace=True)

    # print(df_train)
    # Features on which model will be trained
    features= cols[1:17]
    #features = [cols[1], cols[3], cols[4], cols[5], cols[11], cols[12]]
    # Separating out the feature values
    x_train = df_train.loc[:,features].values
    # Separating out the labels
    y_train = df_train.loc[:,['Category']].values
    # Separating out the test data
    x_test = df_test.loc[:,features].values

    ###### Preprocessing of training and testing set ################
   
    x_train_transposed=np.transpose(x_train)
    x_test_transposed=np.transpose(x_test)
    # x_train=pd.get_dummies(df_train.loc[:,features])
    # x_test=pd.get_dummies(df_test.loc[:,features])

    le = preprocessing.LabelEncoder()
    for i in range(len(features)):
        x_train_transposed[i] = le.fit_transform(x_train_transposed[i])

    for i in range(len(features)):
        x_test_transposed[i] = le.fit_transform(x_test_transposed[i])

    x_train=np.transpose(x_train_transposed)
    x_test=np.transpose(x_test_transposed)

    ###### Standardization of training and testing set ################

    X_scaled_train = preprocessing.scale(x_train)
    X_scaled_test = preprocessing.scale(x_test)

    # print(x_train)

    # ############# Applying PCA to reduce the dimensions #######################

    # pca = PCA(.95)
    # pca.fit(x_train)
    
    # pca_transform_train = pca.transform(x_train)
    # pca_transform_test = pca.transform(x_test)

    
    ############ Random Forest classifier is used to model the training data #################

    rf = RandomForestClassifier()
    rf.fit(X_scaled_train, np.ravel(y_train))
    
    ############ The model obtained from Random Forest classifier is now used to predict the class labels for test data

    predict_values=rf.predict(X_scaled_test)

    result_dict={}

    for i in range(len(predict_values)):
        result_dict[df_test.loc[i][0]]=predict_values[i]

    ######### Output is stored in CSV file #####################################

    with open('result_Q1.csv', 'w') as f:
        for key in result_dict.keys():
            f.write("%s,%s\n"%(key,result_dict[key]))

    df = pd.read_csv("result_Q1.csv", header=None, index_col=None)
    df.columns = ['Application/Permit Number', 'Category']
    df.to_csv("result_Q1.csv", index=False)
    
main()

