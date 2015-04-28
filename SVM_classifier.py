# Built-in packages
import os, sys

# Add-on packages
from pandas import DataFrame as DF
from pandas import read_csv, read_table
from sklearn import svm

######################  command line arguments  ################################
train = sys.argv[1] # tabular dataset
target = sys.argv[2] #tabular dataset
class_column_name = sys.argv[3] #header name for the class column
kernel=sys.argv[4] # choice for kernel function: rbf, linear, poly, sigmoid
degree=int(sys.argv[5]) # (integer) degree for the polynomial kernel, default 3
C=float(sys.argv[6]) # positive regularization parameter
a=float(sys.argv[7]) # positive coefficient in kernel function
b=sys.argv[8] # independent term coefficient in kernel function
outfile1 = sys.argv[9] #output file name (targetset with predicted_class column)
accuracy_on_training = sys.argv[10] #name for string file reporting the accuracy

######################  trainig the SVM  #######################################
train=read_table(train)
classes=train[class_column_name].copy()
del train[class_column_name]
model= svm.SVC(kernel=kernel, C=C, gamma=a, coef0=float(b), degree=degree)
model.fit(train,classes)

##################  predicting classes with the SVM  ###########################
target=read_table(target)
if class_column_name in target.columns:
    del target[class_column_name]
target['predicted_class']=model.predict(target)
target.to_csv(outfile1,index=False,sep='	')

############### computing the accuracy on the training set #####################
train['predicted_class']=model.predict(train)
train[class_column_name]=classes

def correctness(x):
    if x[class_column_name]==x['predicted_class']:
        return 1
    else:
        return 0

def accuracy(data):
    data['correct']=data.apply(correctness,axis=1)
    accuracy=float(data['correct'].sum())/data.shape[0]
    return accuracy

accuracy=str(accuracy(train)*100)+' percent'
os.system("echo %s > %s"%(accuracy,accuracy_on_training))
