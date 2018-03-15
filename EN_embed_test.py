import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import numpy as np
from collections import defaultdict
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


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


def read_corpus(corpus_file):
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
			for line in f:
				try:
					splittedline = line.split('\t')
					documents.append(splittedline[0].split())
					labels.append(splittedline[1].strip('\n').split())
					
				except:
					continue
					
	print("read corpus")
	return documents, labels
	
def ExtraTreesClf(w2v, X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	print('splitted')
	data = TfidfEmbeddingVectorizer(w2v).fit(X_train, y_train)
	print('TFIDF fitted')
	X_train_transformed = data.transform(X_train)
	print('transformed training')
	train_y = np.asarray(y_train)
	y_train = np.array(train_y.ravel())
	#classifier= RandomForestClassifier(n_estimators=100).fit(X_train_transformed , y_train)
	classifier= ExtraTreesClassifier(n_estimators=200).fit(X_train_transformed , y_train)
	print('classifier fitted')
	X_test_transformed = data.transform(X_test)
	test_y = np.asarray(y_test)
	y_test = np.array(test_y.ravel())
	
	return cross_val_score(classifier, X_test_transformed, y_test, cv = 5)

	

def modeltrainer():
	X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
	model = Word2Vec(X, min_count=5, window=40, sg=1)
	model.save('embed_EN.txt')

def main():
	#modeltrainer() ### Execute to create model the first time
	model = Word2Vec.load('embed_EN.txt') ### Loads the model when it has been created
	w2v = dict(zip(model.wv.index2word, model.wv.syn0))
	#with open("glove.6B.100d.txt", "rb") as lines:
		#w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
	X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
	resultEn = ExtraTreesClf(w2v, X, Y)
	print("Accuracy English: " + str(resultEn))	
if __name__ == "__main__":
	main()	
