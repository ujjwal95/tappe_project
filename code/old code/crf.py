import pandas as pd 
import numpy 
import matplotlib
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split 
from sklearn.externals import joblib
from glob import glob
import operator
import sys
import os

# FBcols = ["FB%d" % d for d in range(4096)]
# GGcols = ["GG%d" % d for d in range(512)]
# elmocols = ["ELMO%d" % d for d in range(1024)]

features = ["GS%d" % d for d in range(4096)] + ['wordCount','chartStart','charEnd']

# labelNames = ['semanticType','Symptom','PMH','MEDS','ALLG','FAMHx','SOCHx','pysch','lifestyle','substanceUse','PE','FORM','supportProvision','transition']
labelNames = ['supportProvision']

files = glob("/Users/karanjani/Desktop/csvWithVecs/TrainCSV_Updated/*.csv")

#MAYBE CREATE A LIST FOR featurelabels so you can add what you wish to the FB vectors? 

for name in labelNames:

	featureMaster = []
	labelMaster = []

	for file in files:
		df = pd.read_csv(file)
		df = df.dropna(axis=0, how='any')
		df = df[df.speakerID == 'doctor']
		#DROP ALL LABELS + ANY FEATURES YOU DON'T WANT TO INCLUDE
		dfX = df[features]
		# dfX = df.drop(['labelType','stringList','transition'], axis=1)
		#CREATE LIST OF LIST OF DICTS OF FEATURES
		list_of_FeatureDicts = dfX.to_dict(orient='records')
		featureMaster += [list_of_FeatureDicts]
		#CREATE LIST OF LIST OF STRINGS OF LABELS
		labels = df[name].values.tolist()
		labelMaster += [labels]



	X_train, X_valid, Y_train, Y_valid = train_test_split(featureMaster, labelMaster, test_size = 0)

	crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs',
	c1= 0.1,
	c2=0.1,
	max_iterations=200,
	all_possible_transitions=True)

	
	crf.fit(X_train, Y_train)

	_ = joblib.dump(crf,name+'CRFoptimized.joblib.pkl',compress=9)


