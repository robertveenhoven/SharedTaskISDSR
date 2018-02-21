import sys
import xml.etree.ElementTree as ET

def main(argv):
	x=0
	if sys.argv[2]=='training':
		outputEN= open('Traindata/en/traindataEnglish.txt','wt')
		outputSPA= open('Traindata/spa/traindataSpanish.txt','wt')
		outputARAB= open('Traindata/arab/traindataArabic.txt','wt')   
		truthfile= open(sys.argv[1]+'/truth.txt','r')
		for line in truthfile:
			x+=1
			iduser, gender, variety = line.split(':::')
			print(iduser)
			workfile = open(sys.argv[1]+'/'+iduser+'.xml','r')
			tree = ET.parse(workfile)
			root = tree.getroot()
			language=root.attrib['lang']
			if language == 'en':
				for elem in root.findall('documents/document'):
					tokens=elem.text.split()
					outputEN.write(' '.join(tokens)+' ')
				outputEN.write('\t'+gender+'\n')	
			if language == 'spa':	
				for elem in root.findall('documents/document'):
					tokens=elem.text.split()
					outputSPA.write(' '.join(tokens)+' ')
				outputSPA.write('\t'+gender+'\n')			
			if language == 'arab':
				for elem in root.findall('documents/document'):
					tokens=elem.text.split()
					outputARAB.write(' '.join(tokens)+' ')
				outputARAB.write('\t'+gender+'\n')

		outputEN.close()
		#outputSPA.close() werkt niet sinds deze bestanden nog niet bestaan
		#outputARAB.close()
	
if __name__ == "__main__":
	main(sys.argv)	
