import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, LSTM, MaxPool1D, Flatten, Input, Bidirectional, TimeDistributed, Embedding, GlobalMaxPooling1D, GRU, Merge
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.regularizers import Regularizer
from keras.utils.np_utils import to_categorical
import numpy as np
import sys
import xml.etree.ElementTree as ET
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pickle
from sklearn.model_selection import train_test_split



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


def main(argv):
	with open(argv[3], 'rb') as handle:
		t = pickle.load(handle)
	X_test, Y_test = read_corpus(argv[2])
	model = load_model(argv[1], custom_objects={'AttentionWithContext':AttentionWithContext})
	y_test_reshaped = [1 if tmp_y=='male' else 0 for tmp_y in Y_test]
	y_test_reshaped = to_categorical(np.asarray(y_test_reshaped))
	#y_test_reshaped = (np.asarray(y_test_reshaped)).reshape(1,len(y_test_reshaped),1)
	#y_test_reshaped = y_test_reshaped[0]	
	X_test = t.texts_to_sequences(X_test)
	X_test_reshaped = pad_sequences(X_test, maxlen=int(argv[4]), padding='post')
	print('padded')
	loss, accuracy = model.evaluate(X_test_reshaped, y_test_reshaped, verbose=0)
	print(loss,accuracy)

	
if __name__ == "__main__":
	main(sys.argv)		
