from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def read_corpus(corpus_file):
	tknzr = TweetTokenizer()
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
			for line in f:
				try:
					splittedline = line.split('\t')
					tokens = tknzr.tokenize(splittedline[0])
					for idx, token in enumerate(tokens):
						if token[0]=='@':
							tokens[idx] = 'USER'
						elif token[:4]=='http':
							tokens[idx] = 'URL'	
						elif token[:4]== 't.co':
							tokens[idx] = 'URL'	
					nopunc = [char for char in tokens if char not in string.punctuation]
					nopunc = ' '.join(nopunc)
					filtered = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
					filtered= ' '.join(filtered)
					documents.append(filtered)
					#print(len(filtered))
					labels.append(splittedline[1].strip('\n'))
					#print(splittedline[1].strip('\n'))
					
				except:
					continue
					
	print("read corpus")
	print(len(documents),len(labels))
	return documents, labels

def SVMClassifier(X,Y):
	vec = CountVectorizer(tokenizer=identity)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
	data = vec.fit(X_train)
	X_train_transformed = data.transform(X_train)
	classifier= svm.SVC(C=1.0, kernel='rbf').fit(X_train_transformed , y_train)
	X_test_transformed = data.transform(X_test)
	
	return cross_val_score(classifier, X_test_transformed, y_test)


def identity(x):
	return x

# split corpus in train and test
X, Y = read_corpus('Traindata/en/traindataEnglish2018.txt')
resultEn = SVMClassifier(X,Y)
print("Accuracy English: " + str(resultEn))
X, Y = read_corpus('Traindata/spa/traindataSpanish2018.txt')
resultSpa = SVMClassifier(X,Y)
print("Accuracy Spanish: " + str(resultSpa))
X, Y = read_corpus('Traindata/arab/traindataArabic2018.txt')
resultAr = SVMClassifier(X,Y)
print("Accuracy Arabic: " + str(resultAr))

# test
#Yguess = classifier.predict(X_test_transformed)
#print(f1_score(y_test, Yguess, average="macro"))
#print(precision_score(y_test, Yguess, average="macro"))
#print(recall_score(y_test, Yguess, average="macro"))  
# evaluate
#print(accuracy_score(y_test, Yguess))

