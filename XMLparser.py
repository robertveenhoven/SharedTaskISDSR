import sys
import xml.etree.ElementTree as ET

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
				workfile = open(sys.argv[1]+item+'/text/'+iduser+'.xml','r')
				tree = ET.parse(workfile)
				root = tree.getroot()
				language=root.attrib['lang']
				if item == 'en':
					x+=1
					for elem in root.findall('documents/document'):
						tokens=elem.text.split()
						outputEN.write(' '.join(tokens)+' ')
					outputEN.write('\t'+gender)	
				if item == 'es':	
					for elem in root.findall('documents/document'):
						tokens=elem.text.split()
						outputSPA.write(' '.join(tokens)+' ')
					outputSPA.write('\t'+gender)			
				if item == 'ar':
					for elem in root.findall('documents/document'):
						tokens=elem.text.split()
						outputARAB.write(' '.join(tokens)+' ')
					outputARAB.write('\t'+gender)
		print(x)
		outputEN.close()
		outputSPA.close()
		outputARAB.close()
	
if __name__ == "__main__":
	main(sys.argv)	
