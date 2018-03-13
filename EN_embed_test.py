import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
	
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

def modeltrainer():
	X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
	model = Word2Vec(X, min_count=5, window=40, sg=1)
	model.save('embed_EN.txt')

def main():
	#modeltrainer() ### Execute to create model the first time
	model = Word2Vec.load('embed_EN.txt') ### Loads the model when it has been created
	print(model)
	#words = list(model.wv.vocab)
	#print(words)
	#print(model['morning'])	
	print(model.most_similar('party', topn=10))
if __name__ == "__main__":
	main()	
