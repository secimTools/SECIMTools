from pandas import DataFrame as DF
from pandas import read_csv, read_table
from sklearn.ensemble import RandomForestClassifier
import os, sys

infile = sys.argv[1]
class_column_name = sys.argv[2]
number_of_estimators=int(sys.argv[3])
outfile1 = sys.argv[4]
outfile2 = sys.argv[5]

data=read_table(infile)
classes=data[class_column_name].copy()
del data[class_column_name]

model= RandomForestClassifier(n_estimators=number_of_estimators)
model.fit(data,classes)

importance=DF([data.columns, model.feature_importances_]).T.sort(columns=1,ascending=False)
importance.to_csv(outfile2,index=False, header=False,sep='	')

data=data[importance.ix[:,0].tolist()]
selected_data=DF(model.fit_transform(data,classes))
selected_data=selected_data.ix[:,:1]
selected_data.columns=data.columns[:2]
selected_data[class_column_name]=classes
selected_data.to_csv(outfile1,index=False,sep='	')
