import pickle
import re

import nltk
import sentiment


classifier = pickle.load(open('sentiment_model.sav', 'rb'))
# custom_tweet = 'Hi! Just landed after a great flight by @unitedairlines. Amazing Experience and great food'
# custom_tweet = 'good flight, horrible food, bad leg space'
custom_tweet = input("Enter tweet to analyse \n")
custom_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", custom_tweet).split())
custom_tweet = sentiment.remove_stop_words(list(custom_tweet.lower().split()))
custom_tokens = sentiment.lemmatize_sentence(nltk.TweetTokenizer().tokenize(' '.join(custom_tweet)))
# print(custom_tokens)
print(classifier.classify(dict([token, True] for token in custom_tweet)))
