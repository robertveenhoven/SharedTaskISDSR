import sys
import xml.etree.ElementTree as ET
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


def preProcess(text,language):
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
	x=0
	if sys.argv[2]=='training':
		outputEN= open('Traindata/en/traindataEnglish2018.txt','wt')
		outputSPA= open('Traindata/spa/traindataSpanish2018.txt','wt')
		outputARAB= open('Traindata/arab/traindataArabic2018.txt','wt')
		languagelist=["ar","en","es"]
		for item in languagelist:   
			truthfile= open(sys.argv[1]+item+'/'+item+'.txt','r')
			for line in truthfile:
				iduser, gender = line.split(':::')
				print(item,iduser)
				workfile = open(sys.argv[1]+item+'/text/'+iduser+'.xml','r')
				tree = ET.parse(workfile)
				root = tree.getroot()
				language=root.attrib['lang']
				if item == 'en':
					x+=1
					for elem in root.findall('documents/document'):
						processed = preProcess(elem.text, 'english')
						outputEN.write(processed+' ')
					outputEN.write('\t'+gender)	
				if item == 'es':	
					for elem in root.findall('documents/document'):
						processed = preProcess(elem.text, 'spanish')
						outputSPA.write(processed+' ')
					outputSPA.write('\t'+gender)			
				if item == 'ar':
					for elem in root.findall('documents/document'):
						processed = preProcess(elem.text, 'arabic')
						outputARAB.write(processed+' ')
					outputARAB.write('\t'+gender)
		outputEN.close()
		outputSPA.close()
		outputARAB.close()
		
if __name__ == "__main__":
	main(sys.argv)	
