#Date:05/22/2020
#Name: Rajesh Manicavasagam
#Description : One class SVM classification for NSL-KDD data

#from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sb
from sklearn.cluster import KMeans
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt

#clustering
def kmeans(df):
 print('Kmeans')
 Error = []
 for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(df)
    kmeans.fit(df)
    Error.append(kmeans.inertia_)

 plt.plot(range(1, 11), Error)
 plt.title('Elbow method')
 plt.xlabel('No of clusters')
 plt.ylabel('Error')
 plt.savefig('kmeans.png')
 plt.show()

#calculate max,min,mean and std value for a feature
def calculate(df1,feature):
 print('Maximum  is:',df1[feature].max())
 print('Mimnimum is:',df1[feature].min())
 print('Mean is:',df1[feature].mean())
 print('std  is:',df1[feature].std())
 
def main():
 nRowsRead = 10 # specify 'None' if want to read whole file
# KDD training data 125974
 train_X = pd.read_csv('NIDS\KDDAll.txt', delimiter=',', nrows = 125974)
 train_X = train_X[['duration','protocol_type','service','src_bytes','dst_bytes','num_failed_logins','serror_rate','srv_serror_rate','rerror_rate',
 'srv_rerror_rate','dst_host_serror_rate','dst_host_srv_serror_rate' ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','label']]
 le = preprocessing.LabelEncoder()
 train_X = train_X.apply(le.fit_transform)
 #print('\nTraining data\n')
 #print(train_X.head(5))

 
 print('Kmeans -- started')
 kmeans(train_X)
 print('Kmeans - completed')

if __name__== "__main__":
  main()