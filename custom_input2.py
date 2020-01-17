import pandas as pd
import pickle
import re

import nltk
import sentiment


classifier = pickle.load(open('sentiment_model.sav', 'rb'))
# custom_tweet = 'Hi! Just landed after a great flight by @unitedairlines. Amazing Experience and great food'
# custom_tweet = 'good flight, horrible food, bad leg space'
path = input('Enter the file name: ')
tweets = pd.read_csv(path)
tweets_list = list(tweets['text'])

i = 0
for tweet in tweets_list:
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    tweets_list[i] = tweet.lower()
    i = i + 1
tweets_list = sentiment.remove_stop_words(tweets_list)

tokenizer = nltk.TweetTokenizer()
new_tweets_list = []
for tweet in tweets_list:
    token = tokenizer.tokenize(tweet)
    new_tweets_list.append(sentiment.lemmatize_sentence(token))

sentimentList = []
for tweet in new_tweets_list:
    sentimentList.append(classifier.classify(dict([token, True] for token in tweet)))

tweets['airline_sentiment'] = sentimentList

print(tweets)
tweets.to_csv('Final Tweets with Sentiment.csv', index=False)
