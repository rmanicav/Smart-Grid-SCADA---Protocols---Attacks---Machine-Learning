#Date:05/22/2020
#Name: Rajesh Manicavasagam
#Description : One class SVM classification for NSL-KDD data

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
from sklearn import svm
import random
from sklearn import preprocessing

 
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

 train_X = train_X[train_X.label == 'normal'] 
 count = train_X.size
 print('Count of normal',count)
 le = preprocessing.LabelEncoder()
 train_X = train_X.apply(le.fit_transform)
 print('\nTraining data\n')
 print(train_X.head(5))


 print('\nTesting data\n')
 test_X = pd.read_csv('NIDS\KDDAll.txt', delimiter=',', nrows = 125974)

 test_X = test_X[['duration','protocol_type','service','src_bytes','dst_bytes','num_failed_logins','serror_rate','srv_serror_rate','rerror_rate',
 'srv_rerror_rate','dst_host_serror_rate','dst_host_srv_serror_rate' ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','label']]
 aly = test_X[test_X.label == 'anomaly'].size 
 norm  = test_X[test_X.label == 'normal'].size
 le = preprocessing.LabelEncoder()
 test_X = test_X.apply(le.fit_transform)
 print('Anomaly count',aly)
 print('Normal count',norm)
 #print(test_X.head(5))

 #OCSVM

 print('OCSVM - start')
 print('OCSVM -- completed')

 

if __name__== "__main__":
  main()
