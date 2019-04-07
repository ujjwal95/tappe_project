from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split 
import pandas as pd
from numpy import array
import numpy
from glob import glob

seed = 7
numpy.random.seed(seed)

# class_weights = {0: 0.1, 1: 50000.}
files = glob('/Users/karanjani/Desktop/csvWithVecs/TrainCSV/*.csv')
features = ["GS%d" % d for d in range(4096)] + ['wordCount','chartStart','charEnd']

# Create Model
model = Sequential()
model.add(Conv1D(64, 5, input_shape=(len(features),1), activation='relu'))
model.add(MaxPooling1D(pool_size=4))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# Compile model
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #clippingGrads to avoid explodingGrad
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


for file in files:
	print file
	df = pd.read_csv(file)
	df = df.dropna(axis=0, how='any')
	df = df[df.speakerID == 'doctor'] #only pulls the doctor utterances
	# split into input (X) and output (Y) variables
	dfX = df[features]
	X = array(dfX.values)
	X = numpy.expand_dims(X, axis=2)

	Y = array((df['PMH'] == 'yes').astype(int)) #for binary variables
	# Y = array(pd.get_dummies(df[['supportProvision']])) #for categorical variables

	# Weight classes to create a balanced dataset
	class_weights = class_weight.compute_class_weight('balanced',numpy.unique(Y),Y)

	# Fit the model
	model.fit(X, Y, epochs=10, batch_size=1,class_weight=class_weights)

model.save('PMHCNNv2_hiddenLayers.h5')


