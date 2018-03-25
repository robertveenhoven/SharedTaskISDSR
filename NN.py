# ===================================================================================
# Neural Network script for shared task
# -----------------------------------------------------------------------------------
# Instructions: Ensure Traindata as result of XMLparser AND embedding data
#				is present before executing.
#               Execute without arguments (python NN.py).
#               To chance NN type, adapt classifier names in 'def main()'.
#				
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

#gensim import
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec

from collections import defaultdict

# download stopwords data just in case
import nltk
nltk.download('stopwords')

# -----------------------------------------------------------------------------------
# import for NN
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten, Input
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
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

# -----------------------------------------------------------------------------------
# Embeddings loading
#model = Word2Vec.load('embed_EN.txt') ### Loads the model when it has been created
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))

with open("glove.6B.100d.txt", "rb") as lines:
	w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}


# ===================================================================================
# Pre-processing

# -----------------------------------------------------------------------------------
# TF-IDF embeddings vectorizer (copied from EN_embed_test.py)
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 100 ## for each word the amount of vectors

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def embed_vectorise(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	print('splitted')
	data = TfidfEmbeddingVectorizer(w2v).fit(X_train, y_train)
	print('TFIDF fitted')
	X_train_transformed = data.transform(X_train)
	print('transformed training')
	train_y = np.asarray(y_train)
	y_train = np.array(train_y.ravel())

	X_test_transformed = data.transform(X_test)
	test_y = np.asarray(y_test)
	y_test = np.array(test_y.ravel())

	return (X_train_transformed,y_train)

# -----------------------------------------------------------------------------------
# 	Tfidf vectorisation adapted from BoW.py
# Bag-of Words vectorisation
def bow_vectorise(X,Y):
	vec = TfidfVectorizer(min_df=2, sublinear_tf=True, use_idf =True, ngram_range=(1, 2), preprocessor = identity, tokenizer=identity)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
	data = vec.fit_transform(X_train).toarray() # store vectorisation result as array

	return(data,y_train)

# ===================================================================================
# Neural Network functions


# -----------------------------------------------------------------------------------
# standard feedforward neural network adapted from Rik's NeuralNetwork.py example
def NN(input_dim=None):
	# Model definition
	model=Sequential()
	model.add(Dense(500, input_dim=input_dim, activation="relu"))
	model.add(Dropout(0.01))
	model.add(Dense(500, activation="relu"))
	model.add(Dense(100, activation="relu"))
	model.add(Dense(1, activation="sigmoid"))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

# -----------------------------------------------------------------------------------
# RNN / LSTM classifier
def RNNClassifier(X,Y):
	data, y_train = embed_vectorise(X,Y)

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
	data, y_train = embed_vectorise(X,Y)

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

	# vectorisation
	data, y_train = bow_vectorise(X,Y)
	
	# Data reshaping	
	X_train_reshaped = data # no reshaping necessary
	#convert labels to binary: female -> '0'
	y_train_reshaped = [1 if tmp_y=='male' else 0 for tmp_y in y_train]
	
	# call NN model
	model = KerasClassifier(build_fn=NN, input_dim=len(X_train_reshaped[0]), epochs=20, batch_size=16, verbose=0)
	# Cross-validation
	resultsEn = cross_val_score(model, X_train_reshaped, y_train_reshaped, cv=5)

	print("Accuracy English: " + str(resultsEn.mean()))
	#X, Y = read_corpus('Traindata/spa/traindataSpanish2018.txt')
	#resultSpa = NNClassifier(X,Y)
	#print("Accuracy Spanish: " + str(resultSpa))
	#X, Y = read_corpus('Traindata/arab/traindataArabic2018.txt')
	#resultAr = NNClassifier(X,Y)
	#print("Accuracy Arabic: " + str(resultAr))

if __name__ == "__main__":
	main()	
