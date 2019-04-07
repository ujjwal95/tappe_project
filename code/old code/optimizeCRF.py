import pandas as pd 
import numpy 
import matplotlib
import sklearn_crfsuite
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn.externals import joblib
from glob import glob
import scipy.stats
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

	for file in files[:10]:
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

	X_train, X_valid, Y_train, Y_valid = train_test_split(featureMaster, labelMaster, test_size = 0.2)

	crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True)

	params_space = {'c1': scipy.stats.expon(scale=0.5),'c2': scipy.stats.expon(scale=0.05)}
	f1_scorer = make_scorer(metrics.flat_f1_score,average='weighted', labels=numpy.unique(name))


	rs = RandomizedSearchCV(crf, params_space,
                        cv=2,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=10,
                        scoring=f1_scorer)

	rs.fit(X_train, Y_train)

	print('best params:', rs.best_params_)
	print('best CV score:', rs.best_score_)
	print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))




