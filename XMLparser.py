import sys
import xml.etree.ElementTree as ET
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from googletrans import Translator


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
   
def preProcess(text,language, dictionary):
	x=0
	tknzr = TweetTokenizer()
	tokens = tknzr.tokenize(text)
	nopunc = [char for char in tokens if char not in string.punctuation]
	filtered = [word.lower() for word in ' '.join(nopunc).split() if word.lower() not in stopwords.words(language)]
	for idx, token in enumerate(filtered):
		try:
			if token == 'abcdea':
				key = 'ABCDEA'+str(filtered[idx+1])
				try:
					filtered[idx] = dictionary[key]
					del filtered[idx+1]
				except:
					key = 'ABCDEA'+str(filtered[idx+1]+str(filtered[idx+2]))
					filtered[idx] = dictionary[key]
					del filtered[idx+1]
					del filtered[idx+2]
			if token == 'abcdeb':
				key = 'ABCDEB'+str(filtered[idx+1])
				try:
					filtered[idx] = dictionary[key]
					del filtered[idx+1]
				except:
					key = 'ABCDEB'+str(filtered[idx+1]+str(filtered[idx+2]))
					filtered[idx] = dictionary[key]
					del filtered[idx+1]
					del filtered[idx+2]
									
			if token == 'abcdec':
				key = 'ABCDEC'+str(filtered[idx+1])
				try:
					filtered[idx] = dictionary[key]
					del filtered[idx+1]				
				except:
					key = 'ABCDEC'+str(filtered[idx+1]+str(filtered[idx+2]))
					filtered[idx] = dictionary[key]
					del filtered[idx+1]
					del filtered[idx+2]
									
			if token == "user":
				filtered[idx] = 'USER'
			elif token =='url':
				filtered[idx] = 'URL'
		except:
			x+=1
			#print('Keyerror '+str(x))
			continue						
			
	return ' '.join(filtered)
	
def preProcess2(tokens):
	#nopunc = [char for char in tokens if char not in string.punctuation]
	for idx, token in enumerate(tokens):
		if token[0]=='@':
			tokens[idx] = 'USER'
		elif token[:4]=='http':
			tokens[idx] = 'URL'	
		elif token[:4]== 't.co':
			tokens[idx] = 'URL'					
			
	return ' '.join(tokens)	

def main(argv):
	
	if sys.argv[2]=='test':
		InputLan = sys.argv[3]
		if InputLan == 'en':
			processlanguage = 'english'
			outputEN= open('Testdata/developmentEn.txt','wt')
		if InputLan == 'ar':
			processlanguage = 'arabic'
			outputARAB= open('Testdata/developmentAr.txt','wt')
		if InputLan == 'es':
			processlanguage = 'spanish'
			outputSPA= open('Testdata/developmentSpa.txt','wt')

		truthfile= open(sys.argv[1]+'development'+InputLan+'.txt','r')
		x = 0
		for line in truthfile:
			x+=1
			iduser, gender,variety = line.split(':::')
			print(iduser, x)
			workfile = open(sys.argv[1]+iduser+'.xml','r')
			tree = ET.parse(workfile)
			root = tree.getroot()
			language=root.attrib['lang']
			if InputLan == 'en':
				for elem in root.findall('documents/document'):
					processed = preProcess3(elem.text, 'english')
					outputEN.write(processed+' ')
				outputEN.write('\t'+gender+'\n')	
			if InputLan == 'es':	
				for elem in root.findall('documents/document'):
					processed = preProcess3(elem.text, 'spanish')
					outputSPA.write(processed+' ')
				outputSPA.write('\t'+gender+'\n')			
			if InputLan == 'ar':
				for elem in root.findall('documents/document'):
					processed = preProcess3(elem.text, 'arabic')
					outputARAB.write(processed+' ')
				outputARAB.write('\t'+gender+'\n')
		outputEN.close()
		#outputSPA.close()
		#outputARAB.close()

	
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
		
	if sys.argv[2]=='translate':
		tknzr = TweetTokenizer()
		x=0
		InputLan = sys.argv[3]
		OutputLan = sys.argv[4]
		if OutputLan == 'en':
			OutputMap, processlanguage = 'en', 'english'
		if OutputLan == 'ar':
			OutputMap, processlanguage = 'arab', 'arabic'
		if OutputLan == 'es':
			OutputMap, processlanguage = 'spa', 'spanish'
		OutputFile = 'Traindata/'+OutputMap+'/traindata'+InputLan+'to'+OutputLan+'.txt'
		Writefile = open(OutputFile,'wt')  
		truthfile= open(sys.argv[1]+InputLan+'/'+InputLan+'.txt','r')
		for line in truthfile:
			EmojiDict = {}
			wordsUser, wordsUser2, wordsUser3, wordsUser4 = [], [], [], []
			RandomI, RandomI2, RandomI3, RandomI4 = -1, -1, -1, -1	
			x += 1
			iduser, gender = line.split(':::')
			print(x)
			workfile = open(sys.argv[1]+InputLan+'/text/'+iduser+'.xml','r')
			tree = ET.parse(workfile)
			root = tree.getroot()
			language=root.attrib['lang']
			y=0
			for elem in root.findall('documents/document'):
				y+=1
				if y > 75:
					wordsUser4.append(elem.text+" ")		
				elif y > 50 and y <=75:
					wordsUser3.append(elem.text+" ")
				elif y > 25 and y <=50:
					wordsUser2.append(elem.text+" ")					
				else:
					wordsUser.append(elem.text+" ")				
			stringUser = " ".join(wordsUser)
			tokens = tknzr.tokenize(stringUser)
			for token in tokens:
				RandomI += 1
				if len(token) == 1 and token not in ["1","2","3","4","5","6","7","8","9",
					"A","a","B","b","C","c","D","d","E","e","F",
					"f","G","g","H","h","I","i","J","j","K","k","L","l","M","m","N","n","O","o","P","p","Q","q","R","r","S",
					"s","T","t","U","u","V","v","W","w","X","x","Y","y","Z","z","‘","’","’’","‘‘","…","“","”","•","–","¿","ó","¡"] and token not in string.punctuation:			
					EmojiDict.update({"ABCDEA"+str(RandomI):token})
					tokens[RandomI] = "ABCDEA"+str(RandomI)
			TokenizedUser = preProcess2(tokens)
			Translated = Translator().translate(TokenizedUser, src=InputLan, dest=OutputLan)
			#print("1 works")
			stringUser = " ".join(wordsUser2)
			tokens = tknzr.tokenize(stringUser)
			for token in tokens:
				RandomI2 += 1
				if len(token) == 1 and token not in ["1","2","3","4","5","6","7","8","9",
					"A","a","B","b","C","c","D","d","E","e","F",
					"f","G","g","H","h","I","i","J","j","K","k","L","l","M","m","N","n","O","o","P","p","Q","q","R","r","S",
					"s","T","t","U","u","V","v","W","w","X","x","Y","y","Z","z","‘","’","’’","‘‘","…","“","”","•","–","¿","ó","¡"] and token not in string.punctuation:
					#print(RandomI2, token)	
					EmojiDict.update({"ABCDEB"+str(RandomI2):token})
					tokens[RandomI2] = "ABCDEB"+str(RandomI2)
			TokenizedUser2 = preProcess2(tokens)
			Translated2 = Translator().translate(TokenizedUser2, src=InputLan, dest=OutputLan)
			#print("2 works")
			stringUser = " ".join(wordsUser3)
			tokens = tknzr.tokenize(stringUser)
			for token in tokens:
				RandomI3 += 1
				if len(token) == 1 and token not in ["1","2","3","4","5","6","7","8","9",
					"A","a","B","b","C","c","D","d","E","e","F",
					"f","G","g","H","h","I","i","J","j","K","k","L","l","M","m","N","n","O","o","P","p","Q","q","R","r","S",
					"s","T","t","U","u","V","v","W","w","X","x","Y","y","Z","z","‘","’","’’","‘‘","…","“","”","•","–","¿","ó","¡"] and token not in string.punctuation:
					#print(RandomI3, token)
					EmojiDict.update({"ABCDEC"+str(RandomI3):token})
					tokens[RandomI3] = "ABCDEC"+str(RandomI3)
			TokenizedUser3 = preProcess2(tokens)
			#print(TokenizedUser2)
			Translated3 = Translator().translate(TokenizedUser3, src=InputLan, dest=OutputLan)
			#print("3 works")	
			stringUser = " ".join(wordsUser4)
			tokens = tknzr.tokenize(stringUser)
			for token in tokens:
				RandomI4 += 1
				if len(token) == 1 and token not in ["1","2","3","4","5","6","7","8","9",
					"A","a","B","b","C","c","D","d","E","e","F",
					"f","G","g","H","h","I","i","J","j","K","k","L","l","M","m","N","n","O","o","P","p","Q","q","R","r","S",
					"s","T","t","U","u","V","v","W","w","X","x","Y","y","Z","z","‘","’","’’","‘‘","…","“","”","•","–","¿","ó","¡"] and token not in string.punctuation:
					EmojiDict.update({"ABCDE"+str(RandomI4):token})
					tokens[RandomI4] = "ABCDE"+str(RandomI4)
			TokenizedUser4 = preProcess2(tokens)
			#print(TokenizedUser2)
			Translated4 = Translator().translate(TokenizedUser4, src=InputLan, dest=OutputLan)
			#print("4 works")			

			ToProcess = Translated.text +" "+Translated2.text +" "+Translated3.text	+" "+Translated4.text	
			
			processed = preProcess(ToProcess, processlanguage, EmojiDict)
			#print(processed)
			Writefile.write(processed+'\t'+gender)
			workfile.close()	
		Writefile.close()


	
		
if __name__ == "__main__":
	main(sys.argv)	
