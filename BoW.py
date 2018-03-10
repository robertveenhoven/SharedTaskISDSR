from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

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

def SVMClassifier(X,Y):
	vec = TfidfVectorizer(min_df=2, sublinear_tf=True, use_idf =True, ngram_range=(1, 2), preprocessor = identity, tokenizer=identity)
	#vec = CountVectorizer(ngram_range=(1, 2), preprocessor = identity, tokenizer=identity)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	data = vec.fit(X_train)
	X_train_transformed = data.transform(X_train)
	classifier= svm.LinearSVC(C=1.0).fit(X_train_transformed , y_train)
	X_test_transformed = data.transform(X_test)
	
	return cross_val_score(classifier, X_test_transformed, y_test, cv = 5)


def identity(x):
	return x
	
	
def main():
	X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
	resultEn = SVMClassifier(X,Y)
	print("Accuracy English: " + str(resultEn))
	X, Y = read_corpus('Traindata/spa/traindataSpanish2018.txt')
	resultSpa = SVMClassifier(X,Y)
	print("Accuracy Spanish: " + str(resultSpa))
	X, Y = read_corpus('Traindata/arab/traindataArabic2018.txt')
	resultAr = SVMClassifier(X,Y)
	print("Accuracy Arabic: " + str(resultAr))

if __name__ == "__main__":
	main()	

