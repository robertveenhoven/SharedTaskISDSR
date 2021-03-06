from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import sys
sys.setrecursionlimit(10000)

import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, MaxPool1D, Flatten, Input, Bidirectional, TimeDistributed, Embedding, GlobalMaxPooling1D, GRU, Merge
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.regularizers import Regularizer
from keras.utils.np_utils import to_categorical
import numpy as np

class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)

class AttLayer(Layer):
	def __init__(self, **kwargs):
		self.init = initializers.get('normal')
		#self.input_spec = [InputSpec(ndim=3)]
		super(AttLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape)==3
		#self.W = self.init((input_shape[-1],1))
		self.W = self.init((input_shape[-1],))
		#self.input_spec = [InputSpec(shape=input_shape)]
		self.trainable_weights = [self.W]
		super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

	def call(self, x, mask=None):
		eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
		
		ai = K.exp(eij)
		weights = ai/K.expand_dims(K.sum(ai, axis=1),1)
        
		weighted_input = x*K.expand_dims(weights,2)
		return K.sum(weighted_input, axis=1)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[-1])

# Data loading

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
	f.close()						
	print("read corpus")
	return documents, labels
	
def RNNClassifier(X,Y):
	
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
	
	y_train_reshaped = [1 if tmp_y=='male' else 0 for tmp_y in y_train]
	y_train_reshaped = to_categorical(np.asarray(y_train_reshaped))

	t = Tokenizer()
	t.fit_on_texts(X_train)
	vocab_size = len(t.word_index) + 1
	X_train_reshaped = t.texts_to_sequences(X_train)
	max_length = max([len(s.split()) for s in X_train])
	X_train_reshaped = pad_sequences(X_train_reshaped, maxlen=max_length, padding='post')
	
	embeddings_index = dict()
	f = open('glove.twitter.27B.200d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))
	embedding_matrix = np.zeros((vocab_size, 200)) #################################################33
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	
	#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#adam = optimizers.Adam(lr=0.0001)
	
	def get_model():
		embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False) #################
		sequence_input = Input(shape=(max_length,), dtype='int32')
		embedded_sequences = embedding_layer(sequence_input)
		l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
		l_drop = Dropout(0.2)(l_lstm)
		l_att = AttentionWithContext()(l_drop)
		preds = Dense(2, activation='softmax')(l_att)
		model = Model(sequence_input, preds)
		#model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
		#print(model.summary())
		return model
	
	
	###
	model = KerasClassifier(build_fn=get_model, verbose=10)
	batch_size = [10, 20, 32, 50, 64]
	epochs = [10, 20, 30]
	#batch_size = [50,80]
	#epochs = [1]
	param_grid = dict(batch_size=batch_size, epochs=epochs)
	###
	grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose = 10, n_jobs=-1)
	grid_result = grid.fit(X_train_reshaped, y_train_reshaped)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	
def identity(x):
	return x
	
	
def main():
	X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
	loss, accuracy = RNNClassifier(X,Y)
	print(loss, accuracy)
if __name__ == "__main__":
	main()			
