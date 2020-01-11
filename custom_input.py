import sentiment
import nltk
import re
import pickle

classifier = pickle.load(open('sentiment_model.sav', 'rb'))
custom_tweet = input('Enter a tweet: ')
custom_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", custom_tweet).split())
custom_tweet = sentiment.remove_stop_words(list(custom_tweet.lower().split()))
custom_tokens = sentiment.lemmatize_sentence(nltk.TweetTokenizer().tokenize(' '.join(custom_tweet)))
print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))