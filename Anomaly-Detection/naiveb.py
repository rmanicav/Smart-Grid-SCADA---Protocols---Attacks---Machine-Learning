#Date:05/25/2020
#Name: Rajesh Manicavasagam
#Description : Naive Bayes classification for NSL-KDD data

#@attribute 'duration' real
#@attribute 'protocol_type' {'tcp','udp', 'icmp'} 
#@attribute 'service' {'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'} 
#@attribute 'flag' { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' }
#@attribute 'src_bytes' real
#@attribute 'dst_bytes' real
#@attribute 'land' {'0', '1'}
#@attribute 'wrong_fragment' real
#@attribute 'urgent' real
#@attribute 'hot' real
#@attribute 'num_failed_logins' real
#@attribute 'logged_in' {'0', '1'}
#@attribute 'num_compromised' real
#@attribute 'root_shell' real
#@attribute 'su_attempted' real
#@attribute 'num_root' real
#@attribute 'num_file_creations' real
#@attribute 'num_shells' real
#@attribute 'num_access_files' real
#@attribute 'num_outbound_cmds' real
#@attribute 'is_host_login' {'0', '1'}
#@attribute 'is_guest_login' {'0', '1'}
#@attribute 'count' real
#@attribute 'srv_count' real
#@attribute 'serror_rate' real
#@attribute 'srv_serror_rate' real
#@attribute 'rerror_rate' real
#@attribute 'srv_rerror_rate' real
#@attribute 'same_srv_rate' real
#@attribute 'diff_srv_rate' real
#@attribute 'srv_diff_host_rate' real
#@attribute 'dst_host_count' real
#@attribute 'dst_host_srv_count' real
#@attribute 'dst_host_same_srv_rate' real
#@attribute 'dst_host_diff_srv_rate' real
#@attribute 'dst_host_same_src_port_rate' real
#@attribute 'dst_host_srv_diff_host_rate' real
#@attribute 'dst_host_serror_rate' real
#@attribute 'dst_host_srv_serror_rate' real
#@attribute 'dst_host_rerror_rate' real
#@attribute 'dst_host_srv_rerror_rate' real
#@attribute 'class' {'normal', 'anomaly'}
#from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sb
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
 
#calculate max,min,mean and std value for a feature
def calculate(df1,feature):
 print('Maximum  is:',df1[feature].max())
 print('Mimnimum is:',df1[feature].min())
 print('Mean is:',df1[feature].mean())
 print('std  is:',df1[feature].std())

#Naive bayes
def naivebayes(train_X,train_Y,test_X,test_Y):
 print('Naive Bayes')
 gnb = GaussianNB()
 test_pred_Y = gnb.fit(train_X,train_Y).predict(test_X)
 print(confusion_matrix(test_Y,test_pred_Y))
 CM = confusion_matrix(test_Y,test_pred_Y)
 TN = CM[0][0]
 FN = CM[1][0]
 TP = CM[1][1]
 FP = CM[0][1]
 FPR = FP/(FP + TN)
 print("FPR:",FPR)
 print(classification_report(test_Y,test_pred_Y))
 plot_confusion_matrix(gnb,test_X,test_Y)
 plt.show()

 

def main():
 nRowsRead = 10 # specify 'None' if want to read whole file
# KDD training data 125974
 df = pd.read_csv('NIDS\KDDAll.txt', delimiter=',', nrows = 125974)
 df = df[['duration','protocol_type','service','src_bytes','dst_bytes','num_failed_logins','serror_rate','srv_serror_rate','rerror_rate',
 'srv_rerror_rate','dst_host_serror_rate','dst_host_srv_serror_rate' ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','label']]
 le = preprocessing.LabelEncoder()
 df = df.apply(le.fit_transform)
 x = df.drop('label',axis=1)
 y = df['label']

 train_X,test_X,train_Y,test_Y = train_test_split(x,y,test_size=0.25,random_state=40)
 print('NaiveBayes - start')
 naivebayes(train_X,train_Y,test_X,test_Y)
 print('NaiveBayes -- completed')

 

if __name__== "__main__":
  main()
