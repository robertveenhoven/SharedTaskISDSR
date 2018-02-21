#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "851486025138536449-tk6r8OrQsp70CieZYIIfLGRo9VRRgQ8"
access_token_secret = "9G15Z1UR2Vz9BbUJAhiBJ9b1ARcGgEWUI42ugpMs8qNaO"
consumer_key = 	"DYR1A1WLg5cQlMSvJBDfLULTd"
consumer_secret = "sswtAVCxAGYJGvzqaJ9oltOdr9TN5dl0rq0GAAuW7aGL85YS21"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    stream.filter(track=['الله'], languages=["ar"])
