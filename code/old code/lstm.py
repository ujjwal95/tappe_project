from numpy import array
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split 
from glob import glob
import pandas as pd

# path = raw_input("\n\nPlease input the path to directory with all training CSV files e.g. /Users/karanjani/Desktop/: ")
files = glob("/Users/karanjani/Desktop/csvWithVecs/TrainCSV/*.csv")
features = ["GS%d" % d for d in range(4096)] + ['wordCount','chartStart','charEnd']


model = Sequential()
# model.add(LSTM(100, input_shape=(None, 4106), return_sequences=True, dropout=0.5, recurrent_dropout=0.2)) # MAKE SURE YOU CHANGE SHAPE BASED ON DATAFRAME SHAPE
model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(None,4099)))
model.add(Dense(32, activation='relu'))
model.add(TimeDistributed(Dense(1, activation='sigmoid'))) #CHANGE NUMBER OF OUTPUT LAYERS BASED ON NUMBER OF CLASSES IN PRIMARY LIST 

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,clipvalue=0.75,clipnorm=1.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

#TRAIN
for file in files:
	df = pd.read_csv(file)
	df = pd.read_csv(file)
	df = df.dropna(axis=0, how='any')
	df = df[df.speakerID == 'doctor'] 

	# DROP ALL LABELS + ANY FEATURES YOU DO NOT WANT TO INCLUDE
	dfX = df[features]
	data_X = array(dfX.values)

	data_Y = array((df['FAMHx'] == 'yes').astype(int))
	data_X = data_X.reshape(1,data_Y.shape[0],4099) # MAKE SURE YOU CHANGE LAST DIMENSION BASED ON DATAFRAME SHAPE
	data_Y = data_Y.reshape(1,data_Y.shape[0],1)
	model.fit(data_X, data_Y, epochs=10, batch_size=1)

model.save('FAMHxLSTMv1.h5')


