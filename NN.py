# ===================================================================================
# Neural Network script for shared task
# -----------------------------------------------------------------------------------
# Instructions: Ensure Traindata as result of XMLparser is present before executing.
#               Execute without arguments (python NN.py).
#               To chance NN type, adapt classifier names in 'def main()'.
# Note: no NN prameters (such as loss, optimizer or activation functions have been 
# 		optimized yet.
# ===================================================================================

# ===================================================================================
# set up

# -----------------------------------------------------------------------------------
# Sklean import
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# download stopwords data just in case
import nltk
nltk.download('stopwords')

# -----------------------------------------------------------------------------------
# import for NN
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten, Input
import numpy as np

# -----------------------------------------------------------------------------------
# Data loading (copied from BoW.py)
def read_corpus(corpus_file):
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
			for line in f:
				try:
					splittedline = line.split('\t')
					documents.append(splittedline[0])
					labels.append(splittedline[1].strip('\n'))
					
				except:
					continue
					
	print("read corpus")
	return documents, labels

# ===================================================================================
# Neural Network functions
# 	Tfidf vectorisation adapted from BoW.py

# -----------------------------------------------------------------------------------
# standard feedforward neural network adapted from Rik's NeuralNetwork.py example
def NNClassifier(X,Y):
	# vectorisation
	vec = TfidfVectorizer(min_df=2, sublinear_tf=True, use_idf =True, ngram_range=(1, 2), preprocessor = identity, tokenizer=identity)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
	data = vec.fit_transform(X_train).toarray() # store vectorisation result as array
	
	# Data reshaping	
	X_train_reshaped=data # no reshaping necessary
	#convert labels to binary: female -> '0'
	y_train_reshaped = [1 if tmp_y=='male' else 0 for tmp_y in y_train]

	# Model definition
	model=Sequential()
	model.add(Dense(256, input_dim=len(X_train_reshaped[0]), activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(128, activation="relu"))
	model.add(Dense(1, activation="softmax"))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train_reshaped, y_train_reshaped, batch_size=16, epochs=20, validation_split=0.1, verbose=0)
	predictions = model.predict(X_train_reshaped, batch_size=16, verbose=0)

	return predictions

# -----------------------------------------------------------------------------------
# RNN / LSTM classifier
def RNNClassifier(X,Y):
	vec = TfidfVectorizer(min_df=2, sublinear_tf=True, use_idf =True, ngram_range=(1, 2), preprocessor = identity, tokenizer=identity)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
	data = vec.fit_transform(X_train).toarray() # store vectorisation result as array

	# Data reshaping
	# 	LSTM input_shape requires 3-dimensional input: (Samples, Time steps, Features)
	X_train_reshaped = data.reshape(1,data.shape[0], data.shape[1])
	y_train_reshaped = [1 if tmp_y=='male' else 0 for tmp_y in y_train]
	y_train_reshaped = (np.asarray(y_train_reshaped)).reshape(1,len(y_train_reshaped),1)

	# Model definition
	model = Sequential()
	model.add(LSTM(100, input_shape=X_train_reshaped.shape[1:3],return_sequences=True))
	model.add(Dense(1, activation= 'sigmoid'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.fit(X_train_reshaped, y_train_reshaped, batch_size=16, epochs=20, validation_split=0.1, verbose=0)
	predictions = model.predict(X_train_reshaped, batch_size=16, verbose=0)

	return predictions

# -----------------------------------------------------------------------------------
# Convolutional Neural Network classifier
def CNNClassifier(X,Y):
	vec = TfidfVectorizer(min_df=2, sublinear_tf=True, use_idf =True, ngram_range=(1, 2), preprocessor = identity, tokenizer=identity)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
	data = vec.fit_transform(X_train).toarray() # store vectorisation result as array

	# Reshape data
	#	CNN input should be shaped (max length/ n samples, input length, spatial dimension)
	X_train_reshaped = data.reshape(data.shape[0], data.shape[1],1)
	y_train_reshaped = [1 if tmp_y=='male' else 0 for tmp_y in y_train]
	y_train_reshaped = (np.asarray(y_train_reshaped)).reshape(len(y_train_reshaped),1)
	
	# Model definition
	#	only working CNN method found used 'functional' model, hence the difference 
	#	with the other 2 methods that used the 'sequential' model.
	inp =  Input(shape=X_train_reshaped.shape[1:3])
	conv = Conv1D(filters=2, kernel_size=2)(inp) # kernel size is the 'moving window'
	pool = MaxPool1D(pool_size=2)(conv)
	flat = Flatten()(pool)
	dense = Dense(1)(flat)
	model = Model(inp, dense)

	model.compile(loss='mse', optimizer='adam')

	model.fit(X_train_reshaped, y_train_reshaped)
	predictions = model.predict(X_train_reshaped, batch_size=16, verbose=0)

	return predictions

# -----------------------------------------------------------------------------------
def identity(x):
	return x
	
# ===================================================================================
# script execution
# 	adapt resultEn statements to change NN method used
def main():
	X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
	resultEn = CNNClassifier(X,Y)
	print("Accuracy English: " + str(resultEn))
	X, Y = read_corpus('Traindata/spa/traindataSpanish2018.txt')
	resultSpa = CNNClassifier(X,Y)
	print("Accuracy Spanish: " + str(resultSpa))
	X, Y = read_corpus('Traindata/arab/traindataArabic2018.txt')
	resultAr = CNNClassifier(X,Y)
	print("Accuracy Arabic: " + str(resultAr))

if __name__ == "__main__":
	main()	
