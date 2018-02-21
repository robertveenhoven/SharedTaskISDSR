import json
import sys

def main(argv):

	tweets_file = open(sys.argv[1], "r")
	outputfile= open(sys.argv[2], "a")
	for line in tweets_file:
		try:
			tweet = json.loads(line)
			text_tweets = json.dumps(tweet['text'])
			outputfile.write(text_tweets+'\n')
		except:
			continue	
	outputfile.close()
	
if __name__ == "__main__":
	main(sys.argv)			
