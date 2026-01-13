#Date:05/22/2020
#Name: Rajesh Manicavasagam
#Description : One class SVM classification for NSL-KDD data

#from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

#clustering
def KNN(train_X,train_Y,test_X,test_Y):
 print('KNN')
 knn = KNeighborsClassifier(n_neighbors=1)
 knn.fit(train_X,train_Y)
 pred = knn.predict(test_X)
 print(classification_report(test_Y,pred))
 print(confusion_matrix(test_Y,pred))
 CM = confusion_matrix(test_Y,pred)
 TN = CM[0][0]
 FN = CM[1][0]
 TP = CM[1][1]
 FP = CM[0][1]
 FPR = FP/(FP + TN)
 print("FPR:",FPR)
 print(classification_report(test_Y,pred))
 plot_confusion_matrix(knn,test_X,test_Y)
 plt.show()

#calculate max,min,mean and std value for a feature
def calculate(df1,feature):
 print('Maximum  is:',df1[feature].max())
 print('Mimnimum is:',df1[feature].min())
 print('Mean is:',df1[feature].mean())
 print('std  is:',df1[feature].std())
 
def main():
# KDD training data 125974
 df = pd.read_csv('NIDS\KDDAll.txt', delimiter=',', nrows = 125974)
 df = df[['duration','protocol_type','service','src_bytes','dst_bytes','num_failed_logins','serror_rate','srv_serror_rate','rerror_rate',
 'srv_rerror_rate','dst_host_serror_rate','dst_host_srv_serror_rate' ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','label']]
 le = preprocessing.LabelEncoder()
 df = df.apply(le.fit_transform)
 x = df.drop('label',axis=1)
 y = df['label']

 train_X,test_X,train_Y,test_Y = train_test_split(x,y,test_size=0.25,random_state=40)

 
 print('KNN -- started')
 KNN(train_X,train_Y,test_X,test_Y)
 print('KNN - completed')

if __name__== "__main__":
  main()