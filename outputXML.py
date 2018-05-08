import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.regularizers import Regularizer
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from nltk.tokenize import TweetTokenizer
from keras.models import model_from_json
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
import numpy as np
import pickle
import string
import sys
import os
from lxml.etree import tostring
from lxml.builder import E

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

def preProcess3(text,language):
	x=0
	tknzr = TweetTokenizer()
	tokens = tknzr.tokenize(text)
	nopunc = [char for char in tokens if char not in string.punctuation]
	filtered = [word.lower() for word in ' '.join(nopunc).split() if word.lower() not in stopwords.words(language)]
	for idx, token in enumerate(filtered):
		if token[0]=='@':
			filtered[idx] = 'USER'
		elif token[:4]=='http':
			filtered[idx] = 'URL'	
		elif token[:4]== 't.co':
			filtered[idx] = 'URL'	
					
	return ' '.join(filtered)

def main(argv):
	
	### Eerst namen van beste models en pickle files veranderen hieronder!!!
	
	model_EN = load_model("model_EN.h5", custom_objects={'AttentionWithContext':AttentionWithContext}) # !
	model_ES = load_model("model_ES.h5", custom_objects={'AttentionWithContext':AttentionWithContext}) # !
	model_AR = load_model("model_AR.h5", custom_objects={'AttentionWithContext':AttentionWithContext}) # !
	
	wcount = 1
	inputDir = sys.argv[1]
	for x in os.listdir(inputDir):
		workfile = open(os.path.join(inputDir,x),'r')
		tree = ET.parse(workfile)
		root = tree.getroot()
		language=root.attrib['lang']
		if language == 'en':
			textuser = ''
			ENlist = []
			with open('tokenizer_EN.pickle', 'rb') as handle: # !
				t = pickle.load(handle)
			for elem in root.findall('documents/document'):
				processed = preProcess3(elem.text, 'english')
				textuser += processed + ' '
			ENlist.append(textuser)
			ENlist = t.texts_to_sequences(ENlist)
			ENlist_reshaped = pad_sequences(ENlist, maxlen=1745, padding='post')
			score = model_EN.predict(ENlist_reshaped)
			classPredict = np.argmax(score, axis=1)
			if classPredict == [0]:
				entry = "female"
			elif classPredict == [1]:
				entry = "male"
			authorID = x.split('.')[0]
			out = tostring(E.author(id=authorID,
							lang=language,
							gender_txt=entry)).decode()
			parsestring = 'EN_output/result_'+authorID+'.xml'
			if os.path.isdir('EN_output') == True:
				with open(parsestring, 'w') as fwrite:
					fwrite.write(out)
				print("Written", wcount)
				wcount += 1
			else:
				os.makedirs('EN_output')
				with open(parsestring, 'w') as fwrite:
					fwrite.write(out)
				print("Written", wcount)
				wcount += 1
				
		if language == 'es':
			textuser = ''
			ESlist = []
			with open('tokenizer_ES.pickle', 'rb') as handle: # !
				t = pickle.load(handle)
			for elem in root.findall('documents/document'):
				processed = preProcess3(elem.text, 'spanish')
				textuser += processed + ' '
			ESlist.append(textuser)
			ESlist = t.texts_to_sequences(ESlist)
			ESlist_reshaped = pad_sequences(ESlist, maxlen=1898, padding='post')
			score = model_ES.predict(ESlist_reshaped)
			classPredict = np.argmax(score, axis=1)
			if classPredict == [0]:
				entry = "female"
			elif classPredict == [1]:
				entry = "male"
			authorID = x.split('.')[0]
			out = tostring(E.author(id=authorID,
							lang=language,
							gender_txt=entry)).decode()
			parsestring = 'ES_output/result_'+authorID+'.xml'
			if os.path.isdir('ES_output') == True:
				with open(parsestring, 'w') as fwrite:
					fwrite.write(out)
				print("Written", wcount)
				wcount += 1
			else:
				os.makedirs('ES_output')
				with open(parsestring, 'w') as fwrite:
					fwrite.write(out)
				print("Written", wcount)
				wcount += 1
		
		if language == 'ar':
			textuser = ''
			ARlist = []
			with open('tokenizer_AR.pickle', 'rb') as handle: # !
				t = pickle.load(handle)
			for elem in root.findall('documents/document'):
				processed = preProcess3(elem.text, 'arabic')
				textuser += processed + ' '
			ARlist.append(textuser)
			ARlist = t.texts_to_sequences(ARlist)
			ARlist_reshaped = pad_sequences(ARlist, maxlen=4795, padding='post')
			score = model_AR.predict(ARlist_reshaped)
			classPredict = np.argmax(score, axis=1)
			if classPredict == [0]:
				entry = "female"
			elif classPredict == [1]:
				entry = "male"
			authorID = x.split('.')[0]
			out = tostring(E.author(id=authorID,
							lang=language,
							gender_txt=entry)).decode()
			parsestring = 'AR_output/result_'+authorID+'.xml'
			if os.path.isdir('AR_output') == True:
				with open(parsestring, 'w') as fwrite:
					fwrite.write(out)
				print("Written", wcount)
				wcount += 1
			else:
				os.makedirs('AR_output')
				with open(parsestring, 'w') as fwrite:
					fwrite.write(out)
				print("Written", wcount)
				wcount += 1

if __name__ == "__main__":
	main(sys.argv)
