import pandas as pd
import numpy
import operator
import sys
import matplotlib.pyplot as plt  
from sklearn import preprocessing
from sklearn.svm import SVC
from glob import glob
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  

# path = raw_input("\n\nPlease input the path to directory with all CSV files e.g. /Users/karanjani/Desktop/: ")
path = "/Users/karanjani/Desktop/Test/WeightLossCRF/"

files = glob(path+"*.csv")
featureMasterframe = pd.DataFrame()
labelMaster = []

# MUST CREATE NEW MODEL FOR EACH ATTRIBUTE

for file in files:
	df = pd.read_csv(file)
	# PICK YOUR LABEL 
	labels = df['speakerID'].tolist()
	labelMaster += labels
	# DROP ALL LABELS + ANY FEATURES YOU DO NOT WANT TO INCLUDE
	features = df.drop(['stringList','speakerID','semanticType','Symptom','ChiefConcern','PMH','MEDS','ALLG','SOCHx','FAMHx','substanceUse','sexualHistory','PE','FORM','supportProvision','smokingCess','otherProvider','transition','chartStart','charEnd'], axis=1)
	featureMasterframe = featureMasterframe.append(features)


X_train, X_valid, Y_train, Y_valid = train_test_split(featureMasterframe, labelMaster, test_size = 0.2) 

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, Y_train) 
Y_predict = svclassifier.predict(X_valid) 

print confusion_matrix(Y_valid,Y_predict)
print classification_report(Y_valid,Y_predict) 