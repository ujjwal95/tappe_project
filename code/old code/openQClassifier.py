from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split 
import pandas as pd
from numpy import array
import numpy

seed = 7
numpy.random.seed(seed)

df=pd.read_csv('/Users/karanjani/Desktop/Test/openQMetric_weightLoss.csv')
# split into input (X) and output (Y) variables
dfX = df.drop(['stringList','semanticType'], axis=1)
X = array(dfX.values)
Y = array((df['semanticType'] == 'openQuestion').astype(int))
# split training/testing
X, X_test, Y, Y_test = train_test_split(X, Y, test_size = 0.30)
# create model
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

model.save('openQClassifier_v2.h5')


predictions = model.predict(X_test)
Results = pd.DataFrame({'Predictions':predictions.flatten(),'TrueLabels':Y_test.flatten()})

Results.to_csv('openQPredictions_weightLoss.csv',index=False)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
