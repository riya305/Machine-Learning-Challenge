import csv
import datetime
import math
import numpy as np
from sklearn.feature_selection import RFE
from sklearn import datasets, linear_model
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    #Load csv file into training list with CSV module
    train_dataset=list()
    dict_train={}
    with open('second-question/dataset/hm_train.csv') as f:
        reader = csv.reader(f, delimiter = ',')
        cols=next(reader)
        for line in reader:
            if line != []:
                train_dataset.append(line)

    #Load csv file into testing list with CSV module
    test_dataset=list()
    with open('second-question/dataset/hm_test.csv') as f:
        reader = csv.reader(f, delimiter = ',')
        cols1=next(reader)
        for line in reader:
            if line != []:
                test_dataset.append(line)

    df_train = pd.DataFrame(train_dataset,columns=cols)
    df_test = pd.DataFrame(test_dataset,columns=cols1)

    # used cleaned_hm as the feature for training a model
    features = cols[2]

    # Separating out the training set feature values
    x_train = df_train.loc[:,features].values

    # Separating out the labels
    y_train = df_train.loc[:,['predicted_category']].values

    # Separating out the testing set feature values
    x_test = df_test.loc[:,cols1[2]].values

    ###### Preprocessing of training and testing set ################

    #Preprocessing in order to first vectorize counts of words and
    #then tfidf transformer to generate features form texts.
    tfidf = TfidfVectorizer(sublinear_tf = True,min_df = 5,norm = 'l2',encoding = 'latin-1',ngram_range = (1,2),stop_words='english')
    tfidf.fit(x_train)
    tfidf.fit(x_test)

    tfidf_transform_train = tfidf.transform(x_train)
    
    tfidf_transform_test = tfidf.fit_transform(x_test)

    ############ Naive Bayes classifier is used to model the training data #################

    nb = MultinomialNB()
    nb.fit(tfidf_transform_train,y_train)

    ############ The model obtained from Naive Bayes classifier is now used to predict the class labels for test data

    predict_values=nb.predict(tfidf_transform_test)

    result_dict={}

    for i in range(len(predict_values)):
        result_dict[df_test.loc[i][0]]=predict_values[i]

    ######### Output is stored in CSV file #####################################
    
    with open('result_Q2.csv', 'w') as f:
        for key in result_dict.keys():
            f.write("%s,%s\n"%(key,result_dict[key]))

    df = pd.read_csv("result_Q2.csv", header=None, index_col=None)
    df.columns = ['hmid', 'predicted_category']
    df.to_csv("result_Q2.csv", index=False)
    
main()

